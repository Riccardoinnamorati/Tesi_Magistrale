import subprocess
import os
import signal

def launch_instances(script_path, num_instances):
    """
    Launch multiple instances of a Python script in separate Ubuntu terminals
    and manage their lifecycles.

    :param script_path: Path to the Python script to be executed.
    :param num_instances: Number of instances to launch.
    """
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} does not exist.")
        return

    processes = []  # List to keep track of subprocesses

    try:
        for i in range(num_instances):
            # Command to open a new terminal and execute the Python script with an incremental argument
            command = [
                "xterm", "-hold", "-e",
                f"python3 {script_path} {i} {num_instances}"
            ]

            # Launch the process and add it to the list
            process = subprocess.Popen(command)
            processes.append(process)
            print(f"Launched instance {i} in a new terminal.")

        # Wait for all processes to complete
        for process in processes:
            process.wait()

    except KeyboardInterrupt:
        # Handle early stopping (Ctrl+C) by terminating all subprocesses
        for process in processes:
            if process.poll() is None:  # Check if the process is still running
                try:
                    os.kill(process.pid, signal.SIGTERM)  # Gracefully terminate the process
                    print(f"Terminated process with PID {process.pid}.")
                except OSError as e:
                    print(f"Failed to terminate process {process.pid}: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Ensure all remaining processes are terminated
        for process in processes:
            if process.poll() is None:  # Check if the process is still running
                try:
                    os.kill(process.pid, signal.SIGKILL)  # Forcefully kill the process
                    print(f"Forcefully killed process with PID {process.pid}.")
                except OSError as e:
                    print(f"Failed to kill process {process.pid}: {e}")

if __name__ == "__main__":
    # Replace 'your_script.py' with the actual script you want to execute.
    script_to_run = "iMESA.py"

    # Specify the number of instances you want to launch.
    number_of_instances = 5

    launch_instances(script_to_run, number_of_instances)
