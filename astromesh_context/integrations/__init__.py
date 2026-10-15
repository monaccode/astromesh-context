"""Framework adapters: LangChain middleware, PydanticAI history processor and an MCP server.

Each lives in its own module and needs its extra: ``astromesh-context[langchain]``,
``[pydantic-ai]`` or ``[mcp]``. Importing a module without its extra raises MissingExtraError.
"""
