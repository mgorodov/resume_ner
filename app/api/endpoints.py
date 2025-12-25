from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
import time
import json
from pathlib import Path

from app.core.security import create_access_token, verify_password, get_password_hash
from app.database.session import get_db
from app.database import crud
from app.schemas.schemas import *
from app.api.dependencies import get_current_user, get_current_admin, verify_delete_token

router = APIRouter()


# Authentication endpoints
@router.post("/token", response_model=Token)
async def login_for_access_token(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    user = crud.get_user_by_username(db, user_data.username)
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register")
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    existing_user = crud.get_user_by_username(db, user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(user_data.password)
    user = crud.create_user(
        db=db,
        username=user_data.username,
        hashed_password=hashed_password,
        is_admin=user_data.is_admin
    )
    
    return {"message": "User created successfully", "username": user.username}


# Main forward endpoint
@router.post("/forward", response_model=ForwardResponse)
async def forward_endpoint(
    request: Request,
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Get NER service from app state
        ner_service = request.app.state.ner_service
        if not ner_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="NER service not initialized"
            )
        
        # Check input type
        if image:
            # Process image
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File is not an image"
                )
            
            image_bytes = await image.read()
            result = ner_service.process_image(image_bytes, text)
            
            input_size = len(image_bytes)
            input_type = "image"
            
        elif text:
            # Process text
            result = ner_service.process_text(text)
            
            input_size = len(text)
            input_type = "text"
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid input provided"
            )
        
        processing_time = time.time() - start_time
        
        # Save to history
        crud.create_request_history(
            db=db,
            endpoint="/forward",
            method="POST",
            processing_time=processing_time,
            input_size=input_size,
            input_type=input_type,
            status="success" if result["success"] else "error",
            response_data=result,
            user_agent=user_agent,
            ip_address=client_host
        )
        
        # Check if model failed
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Модель не смогла обработать данные"
            )
        
        # Prepare response
        response_data = {
            "success": True,
            "processing_time": processing_time
        }
        
        if "entities" in result:
            response_data["entities"] = result["entities"]
        if "image_result" in result:
            response_data["image_result"] = result["image_result"]
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Save error to history
        try:
            crud.create_request_history(
                db=db,
                endpoint="/forward",
                method="POST",
                processing_time=processing_time,
                input_size=None,
                input_type=None,
                status="bad_request",
                response_data={"error": str(e)},
                user_agent=request.headers.get("user-agent", "unknown"),
                ip_address=request.client.host if request.client else "unknown"
            )
        except:
            pass  # Если не удалось сохранить в историю, продолжаем
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request"
        )


# Training endpoint
@router.post("/train", dependencies=[Depends(get_current_admin)])
async def train_model(
    n_iter: int = Form(10),
    test_size: float = Form(0.1),
    db: Session = Depends(get_db),
    request: Request = None
):
    start_time = time.time()
    
    try:
        from app.ml.model import ResumeNERModel
        
        # Train model
        print("🎯 Обучение модели NER...")
        model = ResumeNERModel()
        nlp = model.train_model(n_iter=n_iter, test_size=test_size)
        
        # Save model
        model_path = Path("models/resume_ner_model")
        model_path.parent.mkdir(exist_ok=True)
        model.save_model(str(model_path))
        
        # Update app state
        from app.ml.ner_service import NERService
        request.app.state.ner_service = NERService(str(model_path))
        
        # Evaluate
        accuracy = model.evaluate()
        
        processing_time = time.time() - start_time
        
        # Log to history
        if request:
            crud.create_request_history(
                db=db,
                endpoint="/train",
                method="POST",
                processing_time=processing_time,
                status="success",
                response_data={"accuracy": float(accuracy), "iterations": n_iter}
            )
        
        return {
            "success": True,
            "message": "Модель успешно обучена",
            "accuracy": accuracy,
            "processing_time": processing_time,
            "model_path": str(model_path)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        if request:
            try:
                crud.create_request_history(
                    db=db,
                    endpoint="/train",
                    method="POST",
                    processing_time=processing_time,
                    status="error",
                    response_data={"error": str(e)}
                )
            except:
                pass
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при обучении модели: {str(e)}"
        )


# History endpoints
@router.get("/history", response_model=List[HistoryResponse])
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_admin)
):
    try:
        history = crud.get_all_history(db, skip=skip, limit=limit)
        return history
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении истории: {str(e)}"
        )


@router.delete("/history")
async def delete_history(
    db: Session = Depends(get_db),
    delete_confirmed: bool = Depends(verify_delete_token),
    current_user: str = Depends(get_current_admin)
):
    try:
        crud.delete_all_history(db)
        return {"message": "History deleted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении истории: {str(e)}"
        )


# Stats endpoint
@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    days: int = 7,
    db: Session = Depends(get_db)
):
    try:
        stats = crud.get_history_stats(db, days=days)
        
        if not stats:
            return StatsResponse(
                avg_processing_time=None,
                p50_processing_time=None,
                p95_processing_time=None,
                p99_processing_time=None,
                avg_input_size=None,
                request_count=0
            )
        
        return StatsResponse(
            avg_processing_time=stats.avg_time,
            p50_processing_time=stats.p50_time,
            p95_processing_time=stats.p95_time,
            p99_processing_time=stats.p99_time,
            avg_input_size=stats.avg_input_size,
            request_count=stats.request_count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при получении статистики: {str(e)}"
        )