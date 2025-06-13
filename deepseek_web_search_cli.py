import asyncio
from openai import OpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import os
from typing import Optional
import json
import traceback
load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exti_stack = AsyncExitStack()
        self.client: OpenAI = OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY'))

    # 初始化mcp server session
    async def connect_to_server(self):
        server_params = StdioServerParameters(
            command='uv',
            args=['run', 'main.py']
        )

        stdio_transport = await self.exti_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        self.session = await self.exti_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        await self.session.initialize()

    async def process_query(self, query: str) -> str:
        # system prompt 定义，约束大模型的行为
        system_prompt = """
        你是一个专业的信息检索专家，擅长从互联网上搜索信息。
        请根据用户的问题，从互联网上搜索相关信息，并返回搜索结果。
        请注意，你必须使用web_search工具来搜索信息。
        请直接使用web_search工具来搜索信息，不要使用其他工具。
        根据用户的问题，从工具返回的结果中收集和整理答案。
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # Ensure all message contents are strings
        for msg in messages:
            if not isinstance(msg.get("content", ""), str):
                if isinstance(msg["content"], list):
                    msg["content"] = "\n".join(str(item) for item in msg["content"])
                else:
                    msg["content"] = str(msg["content"])

         # 获取所有 mcp 服务器 工具列表信息
        response = await self.session.list_tools()
        # 生成function call的描述信息
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "available_tools": tool.inputSchema
                }
            }
            for tool in response.tools
        ]

        # 请求deepseek 模型
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=available_tools,
            stream=False
        )

        # 处理返回信息
        
        content = response.choices[0]

        if content.finish_reason == "tool_calls":
            tool_calls = content.message.tool_calls[0]
            tool_name = tool_calls.function.name
            tool_input = json.loads(tool_calls.function.arguments)

            # 执行工具
            result = await self.session.call_tool(
                name=tool_name,
                arguments=tool_input)
            
            print(f"\n\n[Calling Tool] {tool_name} with input: {tool_input}")
            print(f"\n\n[Tool Result] {result.content[0].text}")
            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_calls.id,
            })
            # 再次请求deepseek 模型
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        else:
            return content.message.content
        

    async def chat_loop(self):
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in ["exit", "quit", "bye"]:
                    print("Bye!")
                    break
                result = await self.process_query(query)
                print(f"\nAnswer: {result}")
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()

    async def close(self):
        await self.exti_stack.aclose()
    
    async def __aenter__(self):
        await self.connect_to_server()
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.exti_stack.aclose()
            

async def main():
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
    finally:
        await client.close()

if __name__ == "__main__":
    import sys
    asyncio.run(main())