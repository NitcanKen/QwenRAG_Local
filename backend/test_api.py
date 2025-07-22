"""
Simple test script to verify FastAPI structure.
"""

import os
import sys

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set environment variables for testing
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test_service_key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test_anon_key")
os.environ.setdefault("SECRET_KEY", "test_secret_key_for_development_only")

try:
    from app.main import app
    from app.core.config import settings
    
    print("SUCCESS: FastAPI application imported successfully!")
    print(f"SUCCESS: Project name: {settings.PROJECT_NAME}")
    print(f"SUCCESS: API prefix: {settings.API_V1_STR}")
    print(f"SUCCESS: Environment: {settings.ENVIRONMENT}")
    
    # Test basic import of endpoints
    from app.api.api_v1.endpoints import customers, analytics, ml, rag
    print("SUCCESS: All endpoint modules imported successfully!")
    
    # Test model imports
    from app.models.customer import Customer, CustomerCreate, CustomerUpdate
    print("SUCCESS: Customer models imported successfully!")
    
    print("\nSUCCESS: FastAPI project structure is working correctly!")
    
except Exception as e:
    print(f"ERROR: Testing FastAPI structure: {e}")
    import traceback
    traceback.print_exc()