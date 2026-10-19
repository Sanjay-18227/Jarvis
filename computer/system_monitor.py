"""
JARVIS System Performance Monitor Subsystem
Fulfills Section 30: System Monitoring.
Inspects real CPU, RAM, Disk, Battery, and process telemetry to answer performance questions
such as "Jarvis, why is my laptop slow?" with real hardware data.
"""

from typing import Dict, Any, List, Tuple
import psutil


class SystemMonitor:
    """
    Live Windows hardware telemetry and performance diagnosis.
    """

    def get_system_vitals(self) -> Dict[str, Any]:
        """Collects current hardware performance metrics."""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("C:")
        battery = psutil.sensors_battery()

        # Top memory consumers
        procs = []
        for p in psutil.process_iter(["name", "memory_percent", "cpu_percent"]):
            try:
                info = p.info
                if info.get("name"):
                    procs.append(info)
            except Exception:
                pass

        procs.sort(key=lambda x: x.get("memory_percent") or 0, reverse=True)
        top_apps = [p["name"] for p in procs[:4] if p.get("name")]

        battery_info = {
            "percent": battery.percent if battery else 100,
            "plugged": battery.power_plugged if battery else True,
        }

        return {
            "cpu_percent": cpu,
            "memory_percent": mem.percent,
            "memory_available_mb": mem.available // (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free // (1024 * 1024 * 1024),
            "battery": battery_info,
            "top_apps": top_apps,
        }

    def diagnose_performance(self) -> Tuple[bool, str]:
        """
        Diagnoses why the laptop is slow based on live telemetry.
        """
        vitals = self.get_system_vitals()
        cpu = vitals["cpu_percent"]
        ram = vitals["memory_percent"]
        top_apps = ", ".join(vitals["top_apps"][:3])

        reasons = []
        if ram > 85:
            reasons.append(f"RAM usage is very high at {ram}% ({vitals['memory_available_mb']}MB available), mainly consumed by {top_apps}")
        if cpu > 80:
            reasons.append(f"CPU load is peaking at {cpu}%")
        if vitals["disk_free_gb"] < 15:
            reasons.append(f"C: drive is running low on free space ({vitals['disk_free_gb']}GB left)")

        if reasons:
            explanation = "Sir, " + "; ".join(reasons) + ". Closing some background browser tabs or heavy apps should smooth things out."
            return True, explanation

        return True, f"Sir, your laptop hardware looks healthy right now: CPU is at {cpu}%, RAM is at {ram}%, and battery is at {vitals['battery']['percent']}%."
