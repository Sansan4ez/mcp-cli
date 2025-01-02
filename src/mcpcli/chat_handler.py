# chat_handler.py
import json
import os
import logging

from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from mcpcli.llm_client import LLMClient
from mcpcli.system_prompt_generator import SystemPromptGenerator
from mcpcli.tools_handler import convert_to_openai_tools, fetch_tools, handle_tool_call
from mcpcli.messages.send_call_tool import send_call_tool

# Configure logging
logger = logging.getLogger(__name__)

async def handle_chat_mode(server_streams, provider="openai", model="gpt-4o-mini", debug=False):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        tools = []
        tool_to_server = {}  # Mapping of tool name to server streams
        for i, (read_stream, write_stream) in enumerate(server_streams):
            server_tools = await fetch_tools(read_stream, write_stream)
            for tool in server_tools:
                tool_name = tool.get('name')
                if tool_name:
                    tools.append(tool)
                    tool_to_server[tool_name] = (read_stream, write_stream)

        if not tools:
            print("[red]No tools available. Exiting chat mode.[/red]")
            return

        system_prompt = generate_system_prompt(tools)
        openai_tools = convert_to_openai_tools(tools)
        conversation_history = [{"role": "system", "content": system_prompt}]

        # Pass the tool_to_server mapping and tools to the conversation processor
        await process_conversation(
            client=None,  # LLMClient will be instantiated within process_conversation
            conversation_history=conversation_history,
            openai_tools=openai_tools,
            tool_to_server=tool_to_server,
            tools=tools,
            debug=debug
        )
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")


async def process_conversation(
    client,
    conversation_history,
    openai_tools,
    tool_to_server,
    tools,
    debug=False
):
    """Process the conversation loop, handling tool calls and responses."""
    # Initialize LLMClient here to ensure it's used within the correct context
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    client = LLMClient(provider=provider, model=model)
    # conversation_history already contains the system prompt

    while True:
        try:
            user_message = Prompt.ask("[bold yellow]>[/bold yellow]").strip()
            if user_message.lower() in ["exit", "quit"]:
                print(Panel("Exiting chat mode.", style="bold red"))
                break

            user_panel_text = user_message if user_message else "[No Message]"
            print(Panel(user_panel_text, style="bold yellow", title="You"))

            conversation_history.append({"role": "user", "content": user_message})
            completion = client.create_completion(
                messages=conversation_history,
                tools=openai_tools,
            )

            response_content = completion.get("response", "No response")
            tool_calls = completion.get("tool_calls", [])

            if tool_calls:
                tool_responses = []  # Collect all tool responses
                # First, add the assistant's message with tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": response_content if response_content else "I'll help you with that.",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in tool_calls
                    ]
                }
                conversation_history.append(assistant_message)

                for tool_call in tool_calls:
                    try:
                        tool_name = tool_call.function.name
                    except AttributeError:
                        print(f"[red]Error: 'function' attribute not found in tool_call: {tool_call}[/red]")
                        continue

                    if not tool_name:
                        print(f"[red]Invalid tool call: {tool_call}[/red]")
                        continue

                    server_stream = tool_to_server.get(tool_name)
                    if not server_stream:
                        print(f"[red]Tool '{tool_name}' not found on any server.[/red]")
                        continue

                    read_stream, write_stream = server_stream
                    try:
                        arguments_str = tool_call.function.arguments or "{}"
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError as e:
                        print(f"[red]Error parsing arguments for tool '{tool_name}': {e}[/red]")
                        arguments = {}
                    except AttributeError:
                        print(f"[red]Error: 'arguments' attribute not found in tool_call.function: {tool_call.function}[/red]")
                        arguments = {}

                    # Display Tool Invocation
                    # Encode arguments as JSON with ensure_ascii=False for proper Unicode display
                    formatted_args = json.dumps(arguments, indent=2, ensure_ascii=False)
                    tool_md = f"**Tool Call:** {tool_name}\n\n```json\n{formatted_args}\n```"
                    print(
                        Panel(
                            Markdown(tool_md), style="bold magenta", title="Tool Invocation"
                        )
                    )

                    # Send the tool call to the appropriate server
                    result = await send_call_tool(tool_name, arguments, read_stream, write_stream)
                    if result.get("isError"):
                        error_msg = result.get("error", "Unknown error")
                        print(f"[red]Error calling tool '{tool_name}': {error_msg}[/red]")
                        tool_responses.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {error_msg}"
                        })
                    else:
                        response_content = result.get("content", "No content")
                        if debug:
                            logger.debug(f"Tool Response for {tool_name}: {response_content}")
                            print(
                                Panel(
                                    Markdown(f"### Tool Response\n\n{response_content}"),
                                    style="green",
                                )
                            )
                        tool_responses.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": response_content
                        })

                # Add tool responses to conversation history
                for tool_response in tool_responses:
                    # Ensure tool response content is not null
                    if tool_response["content"] is None:
                        tool_response["content"] = "No response from tool"
                    conversation_history.append(tool_response)

                # Get LLM's interpretation of the tool responses
                completion = client.create_completion(
                    messages=conversation_history,
                    tools=openai_tools,
                )

                response_content = completion.get("response") or "I processed the tool responses but have nothing specific to add."
                # Display the LLM's response
                assistant_panel_text = response_content if response_content else "[No Response]"
                print(
                    Panel(Markdown(assistant_panel_text), style="bold blue", title="Assistant")
                )
                conversation_history.append({"role": "assistant", "content": response_content})
                continue

            # Assistant panel with Markdown
            assistant_panel_text = response_content if response_content else "[No Response]"
            print(
                Panel(Markdown(assistant_panel_text), style="bold blue", title="Assistant")
            )
            conversation_history.append({"role": "assistant", "content": response_content})

        except Exception as e:
            print(f"[red]Error processing message:[/red] {e}")
            continue


def generate_system_prompt(tools):
    """
    Generate a concise system prompt for the assistant.

    This prompt is internal and not displayed to the user.
    """
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    system_prompt = prompt_generator.generate_prompt(tools_json)
    system_prompt += """

**GENERAL GUIDELINES:**

1. Step-by-step reasoning:
   - Analyze tasks systematically.
   - Break down complex problems into smaller, manageable parts.
   - Verify assumptions at each step to avoid errors.
   - Reflect on results to improve subsequent actions.

2. Effective tool usage:
   - Explore:
     - Identify available information and verify its structure.
     - Check assumptions and understand data relationships.
   - Iterate:
     - Start with simple queries or actions.
     - Build upon successes, adjusting based on observations.
   - Handle errors:
     - Carefully analyze error messages.
     - Use errors as a guide to refine your approach.
     - Document what went wrong and suggest fixes.

3. Clear communication:
   - Explain your reasoning and decisions at each step.
   - Share discoveries transparently with the user.
   - Outline next steps or ask clarifying questions as needed.

EXAMPLES OF BEST PRACTICES:

- Working with databases:
  - Check schema before writing queries.
  - Verify the existence of columns or tables.
  - Start with basic queries and refine based on results.

- Processing data:
  - Validate data formats and handle edge cases.
  - Ensure integrity and correctness of results.

- Accessing resources:
  - Confirm resource availability and permissions.
  - Handle missing or incomplete data gracefully.

REMEMBER:
- Be thorough and systematic.
- Each tool call should have a clear and well-explained purpose.
- Make reasonable assumptions if ambiguous.
- Minimize unnecessary user interactions by providing actionable insights.

EXAMPLES OF ASSUMPTIONS:
- Default sorting (e.g., descending order) if not specified.
- Assume basic user intentions, such as fetching top results by a common metric.
"""
    return system_prompt
