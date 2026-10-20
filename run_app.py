import os
import sys
import socket
import webbrowser
import qrcode
import uvicorn

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def find_available_port(start_port=8000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    return start_port

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = find_available_port(8000)
    os.environ["PORT"] = str(port)

    local_url = f"http://127.0.0.1:{port}"
    network_url = f"http://{local_ip}:{port}"

    print("\n" + "=" * 65)
    print("   🎙️  SPEECH SENTIMENT & EMOTION RECOGNITION AI")
    print("   RAVDESS Speech Acoustic Analysis | 6th Sem Project")
    print("=" * 65)
    print(f"\n🌐 Computer Browser URL : {local_url}")
    print(f"📱 Mobile / Wi-Fi URL   : {network_url}")
    print("\n📲 SCAN THIS QR CODE WITH YOUR PHONE CAMERA TO OPEN:")
    print("-" * 65)
    
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(network_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
    except Exception as e:
        print(f"(QR terminal preview omitted: {e})")
        
    print("-" * 65)
    print("💡 TIP: Make sure your phone and laptop are on the same Wi-Fi/Hotspot.")
    print("Press Ctrl+C anytime to stop the server.\n" + "=" * 65 + "\n")

    # Automatically open on the local machine
    try:
        webbrowser.open(local_url)
    except Exception:
        pass

    # Host on 0.0.0.0 so both the computer and mobile phones can connect
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
