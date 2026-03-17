@echo off
echo Starting Qpyt-UI MCP Server...
cd /d "%~dp0"
call .venv\Scripts\activate
python api\mcp_server.py
pause
