import sounddevice as sd
from rich.console import Console
from rich.table import Table

def list_audio_devices():
    console = Console()
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    
    table = Table(title="Available Audio Devices")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Input Channels", justify="center")
    table.add_column("Output Channels", justify="center")
    table.add_column("Default", justify="center")

    for i, dev in enumerate(devices):
        is_default = "*" if i == default_input else ""
        table.add_row(
            str(i),
            dev['name'],
            str(dev['max_input_channels']),
            str(dev['max_output_channels']),
            is_default
        )

    console.print(table)
    console.print("\n[bold yellow]TIP:[/bold yellow] Set [bold]MIC_SOURCE[/bold] in your [bold].env[/bold] to the desired Index.")

if __name__ == "__main__":
    list_audio_devices()
