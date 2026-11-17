"""One local nonmodal notification; callback errors never alter experiment status."""
from pathlib import Path
import json,subprocess
from checkpoints import atomic_json
def notify_once(out,status,invoke=subprocess.run):
    try:
        with (out/'notification.claim').open('x') as f:f.write(status)
    except FileExistsError:return
    title='PPG study completed' if status=='completed' else 'PPG study stopped'
    script='''[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime] > $null
$xml=New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text>TITLE</text><text>Local results and terminal.json are saved. Check Codex when available.</text></binding></visual></toast>')
$toast=[Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Windows PowerShell').Show($toast)
'''.replace('TITLE',title)
    p=out/'notify_once.ps1';p.write_text(script,encoding='utf-8-sig')
    result={'attempts':1,'display_observed':False}
    try:
        q=invoke(['powershell.exe','-NoProfile','-NonInteractive','-File',str(p)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
        result.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
    except Exception as exc:result['exception']=repr(exc)
    atomic_json(out/'notification.result.json',result)
