import psutil

def find_process_using_file(file_path):
    for proc in psutil.process_iter(attrs=['pid', 'name', 'open_files']):
        try:
            for file in proc.info['open_files'] or []:
                if file.path == file_path:
                    print(f"Process {proc.info['name']} (PID {proc.info['pid']}) is using the file.")
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    print("No process found using the file.")
    return None

find_process_using_file("C:\\Users\\Hxtreme\\AppData\\Local\\Temp\\tmp3k3nk0ab")
