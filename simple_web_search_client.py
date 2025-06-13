import asyncio
from mcp.client.stdio import stdio_client

from mcp import ClientSession, StdioServerParameters

# create stdio server connection parameters
server_params = StdioServerParameters(
    # 服务器命令参数，使用uv运行
    command='uv',
    # 运行参数
    args = ['run', 'main.py'],
)

async def main():
    # 创建stdio 客户端
    async with stdio_client(server_params) as client:
        # 创建客户端会话
        async with ClientSession(client) as session:
            # 初始化session
            await session.initialize()
            # 列出可用的工具
            tool_response = await session.tool_list()
            print(tool_response)
            # 调用工具
            tool_response = await session.tool_call(
                tool_name='web_search',
                tool_input={'query': '查询一下最近的双色球中奖号码'}
            )
            print(tool_response)

if __name__ == '__main__':
    asyncio.run(main())
