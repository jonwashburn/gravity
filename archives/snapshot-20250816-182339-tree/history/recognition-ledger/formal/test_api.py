#!/usr/bin/env python3
"""Test the Anthropic API connection"""

import anthropic
import asyncio
import aiohttp
import os

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")  # Set via environment variable

async def test_api():
    print("Testing Anthropic API...")
    
    # Test with synchronous client first
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'API is working!' and nothing else."}]
        )
        print(f"✅ Sync API test successful: {response.content[0].text}")
        print(f"   Tokens used: {response.usage.total_tokens}")
    except Exception as e:
        print(f"❌ Sync API test failed: {e}")
    
    # Test async
    print("\nTesting async API...")
    async with aiohttp.ClientSession() as session:
        headers = {
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Say 'Async API is working!' and nothing else."}]
        }
        
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data
            ) as response:
                result = await response.json()
                if response.status == 200:
                    print(f"✅ Async API test successful: {result['content'][0]['text']}")
                    print(f"   Tokens used: {result.get('usage', {}).get('total_tokens', 'unknown')}")
                else:
                    print(f"❌ Async API error {response.status}: {result}")
        except Exception as e:
            print(f"❌ Async API test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_api()) 