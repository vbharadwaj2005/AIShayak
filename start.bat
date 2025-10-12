@echo off
echo Starting AI Shayak Platform...
echo.

echo Starting Python Flask Server...
start "Python Server" cmd /k "cd python && python server.py"

echo Waiting for server to start...
timeout /t 3 /nobreak > nul

echo Starting Next.js Frontend...
start "Next.js Frontend" cmd /k "npm run dev"

echo.
echo Both servers are starting...
echo Python Server: http://localhost:5000
echo Next.js Frontend: http://localhost:3000
echo.
echo Press any key to exit...
pause > nul
