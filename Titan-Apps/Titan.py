import os
import sys
import platform
import subprocess
import shutil
import socket
import time
import datetime
import json
import threading

# Dependency check
try:
    import psutil
    from rich.console import Console
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich.live import Live
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from prompt_toolkit import PromptSession
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.shortcuts import clear
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependencies. Please run: pip install rich prompt_toolkit psutil")
    print(f"Details: {e}")
    sys.exit(1)

# Global Configuration
VERSION = "Titan-OS v2.4.0"
USER = os.environ.get('USER') or os.environ.get('USERNAME') or 'admin'
HOSTNAME = socket.gethostname()
ROOT_DIR = os.getcwd()

# Initialize Rich Console
console = Console()

# --------------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------------

def get_header():
    """Returns the styled header grid."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")
    
    title = Text("TITAN TERMINAL", style="bold cyan")
    meta = Text(f"{USER}@{HOSTNAME} | {datetime.datetime.now().strftime('%H:%M:%S')}", style="dim white")
    
    grid.add_row(title, meta)
    return Panel(grid, style="blue", box=box.HEAVY)

def clear_screen():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_splash():
    """Displays the startup splash screen."""
    clear_screen()
    
    splash_text = """
████████╗██╗████████╗ █████╗ ███╗   ██╗
╚══██╔══╝██║╚══██╔══╝██╔══██╗████╗  ██║
   ██║   ██║   ██║   ███████║██╔██╗ ██║
   ██║   ██║   ██║   ██╔══██║██║╚██╗██║
   ██║   ██║   ██║   ██║  ██║██║ ╚████║
   ╚═╝   ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
    """
    console.print(Align.center(Text(splash_text, style="bold cyan")))
    console.print(Align.center(Text("Initializing System Core...", style="dim white")))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task1 = progress.add_task(description="Loading Modules...", total=100)
        while not progress.finished:
            progress.update(task1, advance=2)
            time.sleep(0.02)
            
    console.print(Align.center(Text("System Ready.", style="bold green")))
    time.sleep(0.5)
    clear_screen()

# --------------------------------------------------------------------------------
# MODULES (MODES)
# --------------------------------------------------------------------------------

class Mode:
    """Base class for all application modes."""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def execute(self, session):
        raise NotImplementedError("Execute method must be implemented by subclasses")

class ShellMode(Mode):
    """Standard System Shell Mode."""
    def __init__(self):
        super().__init__("Shell", "Standard command line interface")
        self.completer = WordCompleter([
            'ls', 'dir', 'cd', 'pwd', 'whoami', 'clear', 'cls', 'exit', 'help', 
            'cat', 'echo', 'mkdir', 'rm', 'python', 'git'
        ], ignore_case=True)

    def execute(self, session):
        console.print(Panel(f"[bold green]Entered {self.name} Mode[/]. Type 'back' to return to menu.", border_style="green"))
        
        while True:
            try:
                # Dynamic Prompt
                cwd = os.getcwd()
                # Shorten home path for display
                home = os.path.expanduser("~")
                if cwd.startswith(home):
                    display_path = "~" + cwd[len(home):]
                else:
                    display_path = cwd

                style = Style.from_dict({
                    'user': '#00ff00 bold',
                    'host': '#ffffff',
                    'path': '#00bfff bold',
                })
                
                message = [
                    ('class:user', f"{USER}"),
                    ('class:host', "@titan"),
                    ('class:host', ":"),
                    ('class:path', f"{display_path}"),
                    ('class:host', "$ "),
                ]

                user_input = session.prompt(message, style=style, completer=self.completer).strip()

                if not user_input:
                    continue

                if user_input.lower() == 'back':
                    break

                if user_input.lower() == 'exit':
                    console.print("[red]Shutting down Titan...[/]")
                    sys.exit(0)

                # Custom Handling for cd
                if user_input.startswith("cd "):
                    try:
                        target = user_input[3:].strip()
                        os.chdir(os.path.expanduser(target))
                    except FileNotFoundError:
                        console.print(f"[red]Error:[/red] Directory '{target}' not found.")
                    except PermissionError:
                        console.print(f"[red]Error:[/red] Permission denied.")
                    continue
                
                # Handling for cls/clear
                if user_input.lower() in ['cls', 'clear']:
                    clear_screen()
                    continue

                # Execute system command
                try:
                    # Shell=True allows pipes and redirects
                    subprocess.run(user_input, shell=True, check=False)
                except KeyboardInterrupt:
                    print("^C")
                except Exception as e:
                    console.print(f"[bold red]Execution Error:[/bold red] {e}")

            except KeyboardInterrupt:
                console.print("\n[yellow]Type 'back' to return to menu or 'exit' to quit.[/]")
            except EOFError:
                break

class DashboardMode(Mode):
    """System Monitoring Dashboard."""
    def __init__(self):
        super().__init__("Dashboard", "Real-time system resource monitoring")

    def get_system_data(self):
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU Grid
        cpu_grid = Table.grid(expand=True)
        cpu_grid.add_column()
        cpu_grid.add_row(f"Usage: {cpu_usage}%")
        cpu_grid.add_row(f"Cores: {psutil.cpu_count(logical=True)}")
        cpu_grid.add_row(f"Freq:  {psutil.cpu_freq().current:.0f}Mhz" if psutil.cpu_freq() else "Freq: N/A")

        # Memory Grid
        mem_grid = Table.grid(expand=True)
        mem_grid.add_column()
        mem_grid.add_row(f"Total: {memory.total / (1024**3):.2f} GB")
        mem_grid.add_row(f"Used:  {memory.used / (1024**3):.2f} GB")
        mem_grid.add_row(f"Free:  {memory.available / (1024**3):.2f} GB")
        mem_grid.add_row(f"Perc:  {memory.percent}%")

        # Disk Grid
        disk_grid = Table.grid(expand=True)
        disk_grid.add_column()
        disk_grid.add_row(f"Total: {disk.total / (1024**3):.2f} GB")
        disk_grid.add_row(f"Used:  {disk.used / (1024**3):.2f} GB")
        disk_grid.add_row(f"Free:  {disk.free / (1024**3):.2f} GB")

        layout = Layout()
        layout.split_column(
            Layout(name="upper"),
            Layout(name="lower")
        )
        layout["upper"].split_row(
            Layout(Panel(cpu_grid, title="CPU", border_style="blue")),
            Layout(Panel(mem_grid, title="Memory", border_style="magenta")),
            Layout(Panel(disk_grid, title="Disk", border_style="yellow"))
        )
        
        # Process List (Top 5 by CPU)
        process_table = Table(title="Top Processes", expand=True, box=box.SIMPLE)
        process_table.add_column("PID", justify="right", style="cyan")
        process_table.add_column("Name", style="white")
        process_table.add_column("Status", style="green")
        process_table.add_column("CPU %", justify="right", style="magenta")

        for proc in sorted(psutil.process_iter(['pid', 'name', 'status', 'cpu_percent']), key=lambda p: p.info['cpu_percent'], reverse=True)[:5]:
            process_table.add_row(
                str(proc.info['pid']),
                proc.info['name'],
                proc.info['status'],
                str(proc.info['cpu_percent'])
            )

        layout["lower"].update(Panel(process_table, border_style="white"))
        
        return layout

    def execute(self, session):
        console.print("[yellow]Starting Dashboard... Press Ctrl+C to exit.[/]")
        try:
            with Live(self.get_system_data(), refresh_per_second=2, screen=True) as live:
                while True:
                    time.sleep(0.5)
                    live.update(self.get_system_data())
        except KeyboardInterrupt:
            console.print("[yellow]Dashboard stopped.[/]")

class NetworkMode(Mode):
    """Network Utilities Suite."""
    def __init__(self):
        super().__init__("Network", "Ping, Port Scan, and IP lookup tools")
        self.completer = WordCompleter(['ping', 'scan', 'ip', 'back', 'help'], ignore_case=True)

    def get_ip_info(self):
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        table = Table(title="Network Interfaces", box=box.ROUNDED)
        table.add_column("Interface", style="cyan")
        table.add_column("Address", style="green")
        
        # Standard psutil net_if_addrs
        addrs = psutil.net_if_addrs()
        for nic, nic_addrs in addrs.items():
            for addr in nic_addrs:
                if addr.family == socket.AF_INET:
                    table.add_row(nic, addr.address)
        
        console.print(table)

    def ping_host(self, host):
        console.print(f"[cyan]Pinging {host}...[/]")
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '4', host]
        
        try:
            subprocess.call(command)
        except Exception as e:
            console.print(f"[red]Ping failed: {e}[/]")

    def scan_ports(self, target):
        console.print(f"[cyan]Scanning top ports on {target} (timeout=0.5s)...[/]")
        common_ports = [21, 22, 23, 25, 53, 80, 110, 443, 3306, 3389, 8080]
        
        table = Table(title=f"Scan Results: {target}")
        table.add_column("Port", justify="right")
        table.add_column("Status", justify="center")

        for port in common_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((target, port))
            if result == 0:
                table.add_row(str(port), "[bold green]OPEN[/]")
            else:
                table.add_row(str(port), "[dim red]CLOSED[/]")
            sock.close()
        
        console.print(table)

    def execute(self, session):
        console.print(Panel(f"[bold magenta]Entered {self.name} Mode[/]. Type 'help' for commands.", border_style="magenta"))
        
        while True:
            try:
                user_input = session.prompt("titan(net) > ", completer=self.completer).strip()
                
                if user_input == 'back':
                    break
                
                parts = user_input.split()
                cmd = parts[0].lower() if parts else ""

                if cmd == 'ip':
                    self.get_ip_info()
                elif cmd == 'ping':
                    if len(parts) > 1:
                        self.ping_host(parts[1])
                    else:
                        console.print("[red]Usage: ping <hostname>[/]")
                elif cmd == 'scan':
                    if len(parts) > 1:
                        self.scan_ports(parts[1])
                    else:
                        console.print("[red]Usage: scan <hostname>[/]")
                elif cmd == 'help':
                    console.print("[bold]Commands:[/bold]")
                    console.print("  [cyan]ip[/]         - Show local network interfaces")
                    console.print("  [cyan]ping <host>[/] - Ping a remote host")
                    console.print("  [cyan]scan <host>[/] - Quick port scan of common ports")
                    console.print("  [cyan]back[/]       - Return to main menu")
                elif cmd == '':
                    continue
                else:
                    console.print(f"[red]Unknown command: {cmd}[/]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Type 'back' to return.[/]")

class NotesMode(Mode):
    """Simple Notebook Mode."""
    def __init__(self):
        super().__init__("Notes", "Read and write quick notes")
        self.note_file = os.path.join(ROOT_DIR, "titan_notes.json")
        self.load_notes()

    def load_notes(self):
        if os.path.exists(self.note_file):
            try:
                with open(self.note_file, 'r') as f:
                    self.notes = json.load(f)
            except:
                self.notes = []
        else:
            self.notes = []

    def save_notes(self):
        with open(self.note_file, 'w') as f:
            json.dump(self.notes, f, indent=4)

    def list_notes(self):
        if not self.notes:
            console.print("[dim]No notes found.[/]")
            return
        
        table = Table(title="Saved Notes", box=box.SIMPLE)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Content", style="white")
        table.add_column("Date", style="dim")

        for idx, note in enumerate(self.notes):
            table.add_row(str(idx), note['content'], note['date'])
        
        console.print(table)

    def add_note(self, content):
        self.notes.append({
            "content": content,
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        self.save_notes()
        console.print("[green]Note saved.[/]")

    def delete_note(self, idx):
        try:
            idx = int(idx)
            if 0 <= idx < len(self.notes):
                removed = self.notes.pop(idx)
                self.save_notes()
                console.print(f"[green]Deleted note: {removed['content'][:20]}...[/]")
            else:
                console.print("[red]Invalid ID.[/]")
        except ValueError:
            console.print("[red]ID must be a number.[/]")

    def execute(self, session):
        console.print(Panel(f"[bold yellow]Entered {self.name} Mode[/].", border_style="yellow"))
        
        while True:
            try:
                user_input = session.prompt("titan(notes) > ").strip()
                
                if user_input == 'back':
                    break
                
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower() if parts else ""

                if cmd == 'list':
                    self.list_notes()
                elif cmd == 'add':
                    if len(parts) > 1:
                        self.add_note(parts[1])
                    else:
                        console.print("[red]Usage: add <text>[/]")
                elif cmd == 'del':
                    if len(parts) > 1:
                        self.delete_note(parts[1])
                    else:
                        console.print("[red]Usage: del <id>[/]")
                elif cmd == 'help':
                    console.print("[bold]Commands:[/bold]")
                    console.print("  [cyan]list[/]       - Show all notes")
                    console.print("  [cyan]add <text>[/] - Save a new note")
                    console.print("  [cyan]del <id>[/]   - Delete a note by ID")
                    console.print("  [cyan]back[/]       - Return to main menu")
                elif cmd == '':
                    continue
                else:
                    console.print("[red]Unknown command. Try 'help'.[/]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Type 'back' to return.[/]")

# --------------------------------------------------------------------------------
# MAIN APPLICATION CONTROLLER
# --------------------------------------------------------------------------------

class TitanTerminal:
    def __init__(self):
        self.modes = {
            "1": ShellMode(),
            "2": DashboardMode(),
            "3": NetworkMode(),
            "4": NotesMode()
        }
        self.session = PromptSession()

    def print_menu(self):
        console.print(get_header())
        console.print("\n[bold underline]AVAILABLE MODULES[/bold underline]\n")
        
        grid = Table.grid(expand=True, padding=(0, 2))
        grid.add_column()
        grid.add_column()
        
        # Display modes in a nice list
        for key, mode in self.modes.items():
            grid.add_row(
                f"[bold cyan][{key}][/] [bold white]{mode.name}[/]", 
                f"[dim]{mode.description}[/]"
            )
        
        grid.add_row("", "")
        grid.add_row("[bold cyan][q][/] [bold white]Quit[/]", "[dim]Exit Application[/]")
        
        console.print(Panel(grid, border_style="dim white", title="Main Menu"))

    def run(self):
        show_splash()
        
        while True:
            clear_screen()
            self.print_menu()
            
            try:
                choice = self.session.prompt("\nSelect Module > ").strip().lower()
                
                if choice in self.modes:
                    clear_screen()
                    self.modes[choice].execute(self.session)
                elif choice in ['q', 'quit', 'exit']:
                    console.print("[bold red]Terminating Session...[/]")
                    sys.exit(0)
                else:
                    console.print("[red]Invalid selection.[/]")
                    time.sleep(1)
            
            except KeyboardInterrupt:
                console.print("\n[bold red]Force Exit.[/]")
                sys.exit(0)

# --------------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure terminal size is sufficient
    try:
        if os.get_terminal_size().columns < 60:
            print("Warning: Terminal window is small. UI might look cluttered.")
    except:
        pass

    app = TitanTerminal()
    app.run()
