# MCP Server Testing Guide

This guide provides curl commands to test the Toy MCP Server and verify that the streamable HTTP transport is working correctly.

## Server Information
- **URL**: http://localhost:8901
- **MCP Endpoint**: http://localhost:8901/mcp/rpc
- **Transport**: Streamable HTTP (Server-Sent Events)

## Basic Server Checks

### 1. Server Status Check
```bash
curl -X GET http://localhost:8901/
```
**Expected Response**: May return 404 or empty response (MCP server doesn't serve static content)

### 2. Check if Server is Running
```bash
lsof -i :8901
```
**Expected Response**: Should show the Python process listening on port 8901

## MCP Protocol Testing

### 3. Server Initialization (Works with curl)
```bash
curl -X POST http://localhost:8901/mcp/rpc \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {
        "name": "curl-test",
        "version": "1.0.0"
      }
    }
  }'
```
**Expected Response**: Server-Sent Events (SSE) format with server info and capabilities

### 4. List Available Tools (Requires proper session management)
```bash
curl -X POST http://localhost:8901/mcp/rpc \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
  }'
```
**Expected Response**: May return "Missing session ID" error - this is expected for HTTP Streaming

## Tool Execution Testing

**Note**: HTTP Streaming transport requires proper session management that curl cannot easily provide. The following commands may return "Missing session ID" errors, which is expected behavior.

### 5. Test Add Function (Limited with curl)
```bash
curl -X POST http://localhost:8901/mcp/rpc \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "add",
      "arguments": {
        "a": 5,
        "b": 3
      }
    }
  }'
```
**Expected Response**: May return "Missing session ID" error - this is expected

### 6. Test Magic 8-Ball (Limited with curl)
```bash
curl -X POST http://localhost:8901/mcp/rpc \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "magic_8_ball",
      "arguments": {
        "question": "Will this test work?"
      }
    }
  }'
```
**Expected Response**: May return "Missing session ID" error - this is expected

## Advanced Testing

### 7. Verbose HTTP Details
```bash
curl -v -X POST http://localhost:8901/mcp/rpc \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'
```
**Useful for**: Debugging HTTP headers, redirects, and response details

## Expected Response Formats

### Successful Server Initialization (SSE Format)
```
event: message
data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"Toy MCP Server","version":"0.0.4"}}}
```

### Session Error Response (Expected for tool calls with curl)
```json
{
  "jsonrpc": "2.0",
  "id": "server-error",
  "error": {
    "code": -32600,
    "message": "Bad Request: Missing session ID"
  }
}
```

### Successful Tool Call (SSE Format - when using proper client)
```
event: message
data: {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"8"}]}}
```

## Troubleshooting

### Common Issues:
1. **404 Not Found**: Server not running or wrong port
2. **Connection Refused**: Server not started
3. **"Missing session ID"**: **EXPECTED** for HTTP Streaming with curl - this is normal behavior
4. **"Method not found"**: Incorrect JSON-RPC method name
5. **"Invalid params"**: Incorrect parameter format
6. **"Not Acceptable: Client must accept both application/json and text/event-stream"**: Missing required Accept headers
7. **Timeout**: Long-running tool execution

### Server Status Check:
```bash
ps aux | grep toy_mcp_server
```

### Port Check:
```bash
lsof -i :8901
```

### Start Server:
```bash
cd demo_server
python toy_mcp_server.py
```

## Notes

- The server uses **Streamable HTTP transport** which requires proper Accept headers
- **IMPORTANT**: The Accept header must include both `application/json` and `text/event-stream` for MCP protocol calls
- **HTTP Streaming requires session management**: Tool calls require proper session handling that curl cannot easily provide
- **Server initialization works with curl**: You can test basic connectivity and server info
- **Tool calls are limited with curl**: Due to session management requirements, tool calls will return "Missing session ID" errors when using curl
- Responses are in **Server-Sent Events (SSE)** format for MCP protocol calls
- No health check endpoints - this is a pure MCP server
- The Python `fastmcp.Client` handles session management automatically
- **For full testing, use a proper MCP client**: The Python `fastmcp.Client` is recommended for complete testing
- This is a clean, minimal implementation without web framework complications
- **HTTP Streaming is the recommended transport for MCP**: This implementation follows MCP best practices
