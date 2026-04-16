import subprocess
import time
import sys

def main():
    print("Starting Federated Learning Simulation...")
    # Start the server
    server_process = subprocess.Popen([sys.executable, "fl_server.py"])
    time.sleep(3)  # Give server time to start

    # Start clients
    client_processes = []
    for hospital in ["A", "B", "C"]:
        print(f"Starting Hospital {hospital} Client...")
        p = subprocess.Popen([sys.executable, "fl_client.py", "--hospital", hospital])
        client_processes.append(p)

    try:
        # Wait for server to finish
        server_process.wait()
        for p in client_processes:
            p.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        server_process.terminate()
        for p in client_processes:
            p.terminate()

if __name__ == "__main__":
    main()
