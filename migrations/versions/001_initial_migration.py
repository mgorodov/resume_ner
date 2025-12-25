"""Initial migration

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create request_history table
    op.create_table('request_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('endpoint', sa.String(), nullable=False),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=False),
        sa.Column('input_size', sa.Integer(), nullable=True),
        sa.Column('input_type', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('response_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index(op.f('ix_request_history_id'), 'request_history', ['id'], unique=False)
    op.create_index(op.f('ix_request_history_timestamp'), 'request_history', ['timestamp'], unique=False)
    
    # Create request_stats table
    op.create_table('request_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('endpoint', sa.String(), nullable=False),
        sa.Column('avg_processing_time', sa.Float(), nullable=True),
        sa.Column('p50_processing_time', sa.Float(), nullable=True),
        sa.Column('p95_processing_time', sa.Float(), nullable=True),
        sa.Column('p99_processing_time', sa.Float(), nullable=True),
        sa.Column('avg_input_size', sa.Float(), nullable=True),
        sa.Column('request_count', sa.Integer(), nullable=True, server_default='0'),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index(op.f('ix_request_stats_id'), 'request_stats', ['id'], unique=False)
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_admin', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_table('users')
    
    op.drop_index(op.f('ix_request_stats_id'), table_name='request_stats')
    op.drop_table('request_stats')
    
    op.drop_index(op.f('ix_request_history_timestamp'), table_name='request_history')
    op.drop_index(op.f('ix_request_history_id'), table_name='request_history')
    op.drop_table('request_history')