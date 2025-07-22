"""
Comprehensive test for Chat Interface API with DeepSeek-R1:8b integration.

Tests the complete chat system including:
- Session management
- Streaming chat responses
- WebSocket connections
- RAG integration
- Error handling
"""

import asyncio
import json
import pytest
import websockets
from httpx import AsyncClient
from datetime import datetime
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
WS_URL = "ws://localhost:8000/api/v1"


class ChatIntegrationTester:
    """Comprehensive test suite for chat integration."""
    
    def __init__(self):
        self.session_id: str = None
        self.test_user_id = "test_user_123"
        self.client: AsyncClient = None
    
    async def setup(self):
        """Set up test environment."""
        print("🔧 Setting up chat integration test...")
        self.client = AsyncClient(base_url=BASE_URL)
        
        # Check if services are healthy
        await self._check_service_health()
    
    async def teardown(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        
        # Delete test session if it exists
        if self.session_id:
            try:
                await self.client.delete(f"/rag/chat/session/{self.session_id}")
            except Exception as e:
                print(f"⚠️ Warning: Could not delete session: {e}")
        
        # Close HTTP client
        if self.client:
            await self.client.aclose()
    
    async def _check_service_health(self):
        """Check if all required services are healthy."""
        print("🏥 Checking service health...")
        
        try:
            response = await self.client.get("/rag/chat/health")
            if response.status_code != 200:
                raise Exception(f"Health check failed with status {response.status_code}")
            
            health_data = response.json()
            if health_data.get("status") not in ["healthy", "degraded"]:
                raise Exception(f"Service unhealthy: {health_data}")
            
            print(f"✅ Service health: {health_data.get('status')}")
            
            # Check individual components
            components = health_data.get("components", {})
            for component, status in components.items():
                component_status = status.get("status", "unknown") if isinstance(status, dict) else status
                print(f"  - {component}: {component_status}")
        
        except Exception as e:
            print(f"❌ Service health check failed: {e}")
            raise
    
    async def test_session_management(self):
        """Test chat session creation, retrieval, and deletion."""
        print("\n🔄 Testing session management...")
        
        # Test 1: Create new session
        print("  Testing session creation...")
        create_response = await self.client.post(
            "/rag/chat/session",
            json={
                "user_id": self.test_user_id,
                "custom_settings": {
                    "temperature": 0.7,
                    "max_tokens": 2048
                }
            }
        )
        
        assert create_response.status_code == 200, f"Session creation failed: {create_response.text}"
        
        session_data = create_response.json()
        assert session_data["success"] is True
        assert "session_id" in session_data
        
        self.session_id = session_data["session_id"]
        print(f"  ✅ Session created: {self.session_id}")
        
        # Test 2: Retrieve session
        print("  Testing session retrieval...")
        get_response = await self.client.get(f"/rag/chat/session/{self.session_id}")
        
        assert get_response.status_code == 200, f"Session retrieval failed: {get_response.text}"
        
        session_info = get_response.json()
        assert session_info["success"] is True
        assert session_info["session"]["session_id"] == self.session_id
        assert session_info["session"]["user_id"] == self.test_user_id
        
        print("  ✅ Session retrieved successfully")
        
        # Test 3: List user sessions
        print("  Testing session listing...")
        list_response = await self.client.get(f"/rag/chat/sessions?user_id={self.test_user_id}")
        
        assert list_response.status_code == 200, f"Session listing failed: {list_response.text}"
        
        sessions_list = list_response.json()
        assert sessions_list["success"] is True
        assert len(sessions_list["sessions"]) >= 1
        
        print(f"  ✅ Found {len(sessions_list['sessions'])} sessions for user")
        
        return True
    
    async def test_non_streaming_chat(self):
        """Test non-streaming chat endpoint."""
        print("\n💬 Testing non-streaming chat...")
        
        # Test message about dashboard analytics
        test_message = "What is the current churn rate for fiber optic customers?"
        
        chat_response = await self.client.post(
            "/rag/chat",
            json={
                "message": test_message,
                "session_id": self.session_id,
                "include_dashboard": True,
                "include_documents": True,
                "stream_response": False
            }
        )
        
        assert chat_response.status_code == 200, f"Chat failed: {chat_response.text}"
        
        chat_data = chat_response.json()
        assert chat_data["success"] is True
        assert "response" in chat_data
        assert chat_data["session_id"] == self.session_id
        assert len(chat_data["response"]) > 0
        
        print(f"  ✅ Response received ({len(chat_data['response'])} chars)")
        print(f"  📊 Metadata: {chat_data.get('metadata', {})}")
        
        return True
    
    async def test_streaming_chat(self):
        """Test Server-Sent Events streaming chat."""
        print("\n🌊 Testing streaming chat...")
        
        test_message = "Why are customers churning according to uploaded documents and current analytics?"
        
        # Use httpx streaming
        async with self.client.stream(
            "POST",
            "/rag/chat/stream",
            json={
                "message": test_message,
                "session_id": self.session_id,
                "include_dashboard": True,
                "include_documents": True
            }
        ) as response:
            assert response.status_code == 200, f"Streaming failed: {response.status_code}"
            
            chunks_received = 0
            response_content = ""
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "content" in data:
                            response_content += data["content"]
                            chunks_received += 1
                        
                        if "session_id" in data:
                            assert data["session_id"] == self.session_id
                        
                        # Stop after reasonable amount or completion
                        if chunks_received > 20 or "total_length" in data.get("metadata", {}):
                            break
                    
                    except json.JSONDecodeError:
                        continue
                
                elif line.startswith("event: chat_complete"):
                    print("  🎯 Streaming completed")
                    break
            
            assert chunks_received > 0, "No streaming chunks received"
            assert len(response_content) > 0, "No content in streaming response"
            
            print(f"  ✅ Received {chunks_received} chunks ({len(response_content)} chars)")
        
        return True
    
    async def test_websocket_chat(self):
        """Test WebSocket chat functionality."""
        print("\n🔌 Testing WebSocket chat...")
        
        websocket_url = f"{WS_URL}/rag/chat/ws/{self.session_id}"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                print("  📡 WebSocket connected")
                
                # Wait for connection confirmation
                response = await websocket.recv()
                connection_msg = json.loads(response)
                assert connection_msg["type"] == "connected"
                
                # Send test message
                test_message = {
                    "type": "chat_message",
                    "content": "What are the main reasons for customer churn?",
                    "include_dashboard": True,
                    "include_documents": True
                }
                
                await websocket.send(json.dumps(test_message))
                print("  📤 Message sent")
                
                # Receive responses
                chunks_received = 0
                response_content = ""
                
                while chunks_received < 10:  # Limit for test
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        message = json.loads(response)
                        
                        if message["type"] == "message_received":
                            print("  📨 Message acknowledged")
                        
                        elif message["type"] == "stream_chunk":
                            content = message["data"].get("content", "")
                            response_content += content
                            chunks_received += 1
                        
                        elif message["type"] == "stream_complete":
                            print("  🏁 WebSocket streaming completed")
                            break
                        
                        elif message["type"] == "error":
                            raise Exception(f"WebSocket error: {message['data']}")
                    
                    except asyncio.TimeoutError:
                        print("  ⏰ WebSocket timeout - ending test")
                        break
                
                assert chunks_received > 0, "No WebSocket chunks received"
                print(f"  ✅ WebSocket test completed ({chunks_received} chunks)")
        
        except Exception as e:
            print(f"  ❌ WebSocket test failed: {e}")
            raise
        
        return True
    
    async def test_error_handling(self):
        """Test error handling scenarios."""
        print("\n🚨 Testing error handling...")
        
        # Test 1: Invalid session ID
        print("  Testing invalid session ID...")
        invalid_response = await self.client.post(
            "/rag/chat",
            json={
                "message": "Test message",
                "session_id": "invalid-session-id"
            }
        )
        
        assert invalid_response.status_code == 404, "Should return 404 for invalid session"
        print("  ✅ Invalid session handled correctly")
        
        # Test 2: Empty message
        print("  Testing empty message...")
        empty_response = await self.client.post(
            "/rag/chat",
            json={
                "message": "",
                "session_id": self.session_id
            }
        )
        
        # Should still work (might return a help message)
        assert empty_response.status_code == 200, "Empty message should be handled"
        print("  ✅ Empty message handled")
        
        # Test 3: Session deletion
        print("  Testing session deletion...")
        delete_response = await self.client.delete(f"/rag/chat/session/{self.session_id}")
        
        assert delete_response.status_code == 200, f"Session deletion failed: {delete_response.text}"
        
        delete_data = delete_response.json()
        assert delete_data["success"] is True
        
        print("  ✅ Session deleted successfully")
        
        # Test 4: Access deleted session
        print("  Testing access to deleted session...")
        deleted_response = await self.client.get(f"/rag/chat/session/{self.session_id}")
        
        assert deleted_response.status_code == 404, "Should return 404 for deleted session"
        print("  ✅ Deleted session access handled correctly")
        
        # Clear session ID since it's deleted
        self.session_id = None
        
        return True
    
    async def test_integration_with_rag_system(self):
        """Test integration with unified RAG system."""
        print("\n🧠 Testing RAG system integration...")
        
        # Create a new session for this test
        create_response = await self.client.post(
            "/rag/chat/session",
            json={"user_id": self.test_user_id}
        )
        
        session_data = create_response.json()
        test_session_id = session_data["session_id"]
        
        try:
            # Test dashboard-focused query
            dashboard_query = "Show me the current churn rate breakdown by contract type"
            
            dashboard_response = await self.client.post(
                "/rag/chat",
                json={
                    "message": dashboard_query,
                    "session_id": test_session_id,
                    "include_dashboard": True,
                    "include_documents": False
                }
            )
            
            assert dashboard_response.status_code == 200
            dashboard_data = dashboard_response.json()
            
            # Should have some form of analytics data
            assert len(dashboard_data["response"]) > 0
            print("  ✅ Dashboard-focused query handled")
            
            # Test document-focused query (may not have documents)
            document_query = "What do the uploaded industry reports say about customer retention?"
            
            document_response = await self.client.post(
                "/rag/chat",
                json={
                    "message": document_query,
                    "session_id": test_session_id,
                    "include_dashboard": False,
                    "include_documents": True
                }
            )
            
            assert document_response.status_code == 200
            document_data = document_response.json()
            
            # Should indicate if no documents are available
            assert len(document_data["response"]) > 0
            print("  ✅ Document-focused query handled")
            
            # Test hybrid query
            hybrid_query = "Compare our current churn metrics with industry benchmarks from uploaded reports"
            
            hybrid_response = await self.client.post(
                "/rag/chat",
                json={
                    "message": hybrid_query,
                    "session_id": test_session_id,
                    "include_dashboard": True,
                    "include_documents": True
                }
            )
            
            assert hybrid_response.status_code == 200
            hybrid_data = hybrid_response.json()
            
            assert len(hybrid_data["response"]) > 0
            print("  ✅ Hybrid query handled")
            
            # Check conversation history
            history_response = await self.client.get(f"/rag/chat/session/{test_session_id}")
            history_data = history_response.json()
            
            # Should have at least 6 messages (3 user + 3 assistant)
            conversation_length = len(history_data["conversation_history"])
            assert conversation_length >= 6, f"Expected at least 6 messages, got {conversation_length}"
            
            print(f"  ✅ Conversation history: {conversation_length} messages")
        
        finally:
            # Clean up test session
            await self.client.delete(f"/rag/chat/session/{test_session_id}")
        
        return True
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Chat Interface API Integration Tests")
        print("=" * 60)
        
        test_results = []
        
        try:
            await self.setup()
            
            # Run all test categories
            tests = [
                ("Session Management", self.test_session_management),
                ("Non-Streaming Chat", self.test_non_streaming_chat),
                ("Streaming Chat", self.test_streaming_chat),
                ("WebSocket Chat", self.test_websocket_chat),
                ("RAG Integration", self.test_integration_with_rag_system),
                ("Error Handling", self.test_error_handling),
            ]
            
            for test_name, test_func in tests:
                try:
                    print(f"\n📋 Running {test_name} tests...")
                    result = await test_func()
                    test_results.append((test_name, "PASSED", None))
                    print(f"✅ {test_name}: PASSED")
                
                except Exception as e:
                    test_results.append((test_name, "FAILED", str(e)))
                    print(f"❌ {test_name}: FAILED - {e}")
        
        finally:
            await self.teardown()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🏁 TEST SUMMARY")
        print("=" * 60)
        
        passed_count = 0
        failed_count = 0
        
        for test_name, status, error in test_results:
            if status == "PASSED":
                print(f"✅ {test_name}: PASSED")
                passed_count += 1
            else:
                print(f"❌ {test_name}: FAILED")
                if error:
                    print(f"   Error: {error}")
                failed_count += 1
        
        print("-" * 60)
        print(f"Total Tests: {len(test_results)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/len(test_results)*100):.1f}%")
        
        if failed_count == 0:
            print("\n🎉 All tests passed! Chat interface is working correctly.")
        else:
            print(f"\n⚠️ {failed_count} test(s) failed. Please review the errors above.")
        
        return failed_count == 0


async def main():
    """Main test runner."""
    tester = ChatIntegrationTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())