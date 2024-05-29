#!bin/bash 

# Start the FastAPI app
echo "Starting FastAPI app..."
python3 src/api.py &
FASTAPI_PID=$!
echo "FastAPI app started with PID $FASTAPI_PID"

# start streamlit
echo "Starting Streamlit app..."
streamlit run main.py --server.enableXsrfProtection false &
STREAMLIT_PID=$!
echo "Streamlit app started with PID $STREAMLIT_PID"

# Function to stop both apps
stop_apps() {
    echo "Stopping FastAPI app..."
    kill $FASTAPI_PID
    echo "Stopping Streamlit app..."
    kill $STREAMLIT_PID
}
trap stop_apps EXIT

# Wait for both processes to complete
wait $FASTAPI_PID
wait $STREAMLIT_PID