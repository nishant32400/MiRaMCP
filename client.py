import os
import json
import logging
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tool_registry import TOOLS

# Load environment variables
load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY not set in environment")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("FlightOps.MCPClient")

# Initialize Groq LLM client
client_groq = Groq(api_key=GROQ_API_KEY)

def _build_tool_prompt() -> str:
    """Convert TOOLS dict into compact text to feed the LLM."""
    lines = []
    for name, meta in TOOLS.items():
        arg_str = ", ".join(meta["args"])
        lines.append(f"- {name}({arg_str}): {meta['desc']}")
    return "\n".join(lines)

SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls.
Use only these tools exactly as defined below:

{_build_tool_prompt()}

Rules:
1. Output only valid JSON.
2. Always return a top-level key 'plan' as a list.
3. If user asks something general like 'details of flight', use get_flight_basic_info.
4. Do not invent tool names.
5. If carrier or date not mentioned, omit them instead of writing 'unknown'.
6. only use "tool" as key not "name"
"""

SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise answer.
Focus on clarity and readability.
"""

class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None

    async def connect(self):
        """Connect to the MCP server using streamable-http transport."""
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
            
            # streamablehttp_client returns a context manager
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
            
            # Create session
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            
            # Initialize the connection
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    # ---------- MCP Server Interaction ----------

    async def list_tools(self) -> dict:
        """List available tools from the MCP server."""
        try:
            if not self.session:
                await self.connect()
            
            tools_list = await self.session.list_tools()
            
            # Convert MCP tools response to dictionary format
            tools_dict = {}
            for tool in tools_list.tools:
                tools_dict[tool.name] = {
                    "description": tool.description,
                    "inputSchema": tool.inputSchema
                }
            
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {"error": str(e)}

    async def invoke_tool(self, tool_name: str, args: dict) -> dict:     
        """Invoke a tool by name with arguments via MCP protocol."""
        try:
            if not self.session:
                await self.connect()
            
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            
            # Call the tool using MCP session
            result = await self.session.call_tool(tool_name, args)     
            
            # Extract content from result
            if result.content:
                # MCP returns content as a list of Content objects
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            # Try to parse as JSON
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
                
                # If single item, return it directly
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}
            
            return {"error": "No content in response"}
            
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            return {"error": str(e)}

    # ---------- LLM Wrappers ----------

    def _call_groq(self, messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Internal helper for LLM chat completions."""
        try:
            completion = client_groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return json.dumps({"error": str(e)})

    def plan_tools(self, user_query: str) -> dict:
        """Use LLM to generate a plan of tool calls."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]
        content = self._call_groq(messages, temperature=0.1)
        try:
            plan = json.loads(content)
            if isinstance(plan, dict) and "plan" in plan:
                return plan
            else:
                return {"plan": []}
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM plan output.")
            return {"plan": []}

    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        """Use LLM to summarize results into human-friendly output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question:\n{user_query}"},
            {"role": "assistant", "content": f"Plan:\n{json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Results:\n{json.dumps(results, indent=2)}"},
        ]
        summary = self._call_groq(messages, temperature=0.3)
        return {"summary": summary}

    # ---------- Orchestration ----------

    async def run_query(self, user_query: str) -> dict:
        """
        Full flow:
        1. Use LLM to plan tool calls.
        2. Execute tools sequentially on MCP server.
        3. Summarize results via LLM.
        """
        logger.info(f"User query: {user_query}")
        plan_data = self.plan_tools(user_query)
        plan = plan_data.get("plan", [])

        if not plan:
            return {"error": "LLM did not produce a valid tool plan."}

        results = []
        for step in plan:
            tool = step.get("tool")
            args = step.get("arguments", {})

            # Clean up 'unknown' or empty args
            args = {
                k: v for k, v in args.items()
                if v is not None and str(v).strip() != "" and str(v).lower() != "unknown"
            }

            if not tool:
                continue

            logger.info(f"Invoking tool: {tool} with args: {args}")
            resp = await self.invoke_tool(tool, args)
            results.append({tool: resp})

        # Summarize results
        summary = self.summarize_results(user_query, plan, results)
        return {"plan": plan, "results": results, "summary": summary}