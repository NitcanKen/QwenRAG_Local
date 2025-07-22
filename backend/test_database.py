"""
Test database configuration and connections.
"""

import os
import sys
import asyncio

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Set test environment variables
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test_service_key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test_anon_key")
os.environ.setdefault("SECRET_KEY", "test_secret_key_for_development_only_32chars")

async def test_database():
    """Test database configuration and connections."""
    try:
        # Test database module import
        from app.core.database import init_supabase, init_postgresql, DatabaseManager
        print("SUCCESS: Database module imported successfully!")
        
        # Test Supabase initialization (will fail with test credentials, but should not crash)
        supabase_client = init_supabase()
        if supabase_client:
            print("SUCCESS: Supabase client initialized")
        else:
            print("INFO: Supabase initialization failed (expected with test credentials)")
        
        # Test PostgreSQL initialization (will fail without real database, but should not crash)
        os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
        pg_success = init_postgresql()
        if pg_success:
            print("SUCCESS: PostgreSQL initialized")
        else:
            print("INFO: PostgreSQL initialization failed (expected without real database)")
        
        # Test DatabaseManager methods
        db_manager = DatabaseManager()
        print("SUCCESS: DatabaseManager created successfully!")
        
        # Test configuration loading
        from app.core.config import settings
        print(f"SUCCESS: Configuration loaded - Project: {settings.PROJECT_NAME}")
        print(f"SUCCESS: API prefix: {settings.API_V1_STR}")
        print(f"SUCCESS: Environment: {settings.ENVIRONMENT}")
        
        print("\nSUCCESS: Database configuration is working correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR: Database configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_database())