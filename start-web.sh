#!/bin/bash

# MIOW-Context Web UI Launcher
# Starts both backend and frontend servers
export GEMINI_API_KEY="AIzaSyCjSvRD91M3GZmIK2qHEFu6IqbrdleC1wU"

set -e

echo "🚀 Starting MIOW-Context Web Interface"
echo "═══════════════════════════════════════════════"

# Check if required environment variables are set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY environment variable not set"
    echo "Please set it with: export GEMINI_API_KEY='your_key_here'"
    exit 1
fi

# Check if Qdrant is running
echo "📊 Checking Qdrant..."
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "❌ Qdrant is not running"
    echo "Please start it with: docker-compose up -d"
    exit 1
fi
echo "✅ Qdrant is running"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $backend_pid $frontend_pid 2>/dev/null || true
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔧 Starting Rust backend server..."
cargo run --features web -- serve --port 3001 &
backend_pid=$!

# Wait a moment for backend to start
# Wait for backend to start (up to 60 seconds)
echo "⏳ Waiting for backend to start..."
max_retries=60
counter=0
backend_ready=false

while [ $counter -lt $max_retries ]; do
    if curl -s http://localhost:3001/api/health > /dev/null 2>&1; then
        backend_ready=true
        break
    fi
    sleep 1
    counter=$((counter+1))
    echo -n "."
done
echo ""

if [ "$backend_ready" = false ]; then
    echo "❌ Backend failed to start after $max_retries seconds"
    kill $backend_pid 2>/dev/null || true
    exit 1
fi
echo "✅ Backend running at http://localhost:3001"

# Start frontend server
echo "🎨 Starting React frontend..."
cd web
npm run dev &
frontend_pid=$!

# Wait a moment for frontend to start
sleep 5

echo ""
echo "🎉 Both servers started successfully!"
echo "🌐 Frontend: http://localhost:5173"
echo "🔧 Backend API: http://localhost:3001/api"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for processes
wait
