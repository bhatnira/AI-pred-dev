#!/bin/bash

# ChemML Suite Quick Start Script
# ================================

set -e  # Exit on any error

echo "🧪 ChemML Suite - Quick Start"
echo "=============================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    echo "   📥 Download: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Check if Docker Compose is available  
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Check if we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    echo "❌ docker-compose.yml not found. Please run this script from the project root directory."
    exit 1
fi

echo "✅ Project files found"

# Stop any existing containers
echo "🛑 Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start the application
echo "🚀 Building and starting ChemML Suite..."
echo "   This may take a few minutes on first run..."

if docker-compose up --build -d; then
    echo ""
    echo "🎉 SUCCESS! ChemML Suite is now running!"
    echo ""
    echo "📱 Access your application at:"
    echo "   🌐 http://localhost:8501"
    echo ""
    echo "🔧 Useful commands:"
    echo "   docker-compose logs -f     # View live logs"
    echo "   docker-compose down        # Stop the application"
    echo "   docker-compose restart     # Restart the application"
    echo "   docker ps                  # Check container status"
    echo ""
    echo "🩺 Health check:"
    echo "   The container should show 'healthy' status in 'docker ps'"
    echo ""
    
    # Wait a moment and check if container is healthy
    echo "⏳ Waiting for health check..."
    sleep 10
    
    if docker ps | grep -q "healthy"; then
        echo "✅ Container is healthy and ready!"
        echo ""
        echo "🎯 Ready to explore chemistry with AI!"
        
        # Offer to open browser (macOS only)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            read -p "🌐 Open in browser? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                open http://localhost:8501
            fi
        fi
    else
        echo "⚠️  Container may still be starting up. Check with: docker-compose logs"
    fi
    
else
    echo "❌ Failed to start ChemML Suite. Check the logs with:"
    echo "   docker-compose logs"
    exit 1
fi
