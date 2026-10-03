import os
import time
import subprocess
from playwright.sync_api import sync_playwright

def generate_all_screenshots():
    # Directories
    base_dir = r"E:\CS619\Graze\Prototype\graze_smasher"
    screenshots_dir = os.path.join(base_dir, "Screenshots")
    notebook_html = os.path.join(base_dir, "Notebook", "fake_review_detection.html")
    
    os.makedirs(screenshots_dir, exist_ok=True)
    
    # 1. Start Django Server in background
    django_cwd = os.path.join(base_dir, "Django_Project")
    python_exe = os.path.join(base_dir, ".venv", "Scripts", "python.exe")
    
    print("Starting Django server in background...")
    django_process = subprocess.Popen(
        [python_exe, "manage.py", "runserver", "127.0.0.1:8000"],
        cwd=django_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give the server time to start up
    time.sleep(10)
    
    # Path to local Chrome executable
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    
    print("Launching headless Chrome via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chrome_path,
            headless=True
        )
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        
        # ----------------------------------------------------
        # PART A: Notebook Screenshots (Task 1, 2, 3)
        # ----------------------------------------------------
        print("Navigating to Notebook HTML...")
        page.goto(f"file:///{notebook_html.replace(os.sep, '/')}")
        page.wait_for_timeout(1000)
        
        # Screenshot Task 1: Dataset Collection (showing head output)
        print("Capturing Task 1 (Dataset Collection)...")
        # Find cell that runs df.head() and its output
        task1_locator = page.locator("div.cell").filter(has_text="df.head()")
        if task1_locator.count() > 0:
             task1_locator.first.screenshot(path=os.path.join(screenshots_dir, "dataset_collection.png"))
        else:
            page.screenshot(path=os.path.join(screenshots_dir, "dataset_collection.png"), full_page=True)
            
        # Screenshot Task 2: Data Preprocessing
        print("Capturing Task 2 (Preprocessing)...")
        task2_locator = page.locator("div.cell").filter(has_text="=== ORIGINAL REVIEW ===")
        if task2_locator.count() > 0:
            task2_locator.first.screenshot(path=os.path.join(screenshots_dir, "preprocessing.png"))
        else:
            page.screenshot(path=os.path.join(screenshots_dir, "preprocessing.png"), full_page=True)
            
        # Screenshot Task 3: Model Metrics
        print("Capturing Task 3 (Model Metrics)...")
        task3_locator = page.locator("div.cell").filter(has_text="MODEL METRICS")
        if task3_locator.count() > 0:
            task3_locator.first.screenshot(path=os.path.join(screenshots_dir, "model_metrics.png"))
        else:
            page.screenshot(path=os.path.join(screenshots_dir, "model_metrics.png"), full_page=True)
            
        # ----------------------------------------------------
        # PART B: Django Web Interface Screenshots (Task 4, 5)
        # ----------------------------------------------------
        print("Navigating to Django App Homepage...")
        page.goto("http://127.0.0.1:8000/")
        page.wait_for_timeout(1000)
        
        # Screenshot Task 4: Web Interface Homepage (empty form)
        print("Capturing Task 4 (Web Interface Homepage)...")
        page.screenshot(path=os.path.join(screenshots_dir, "web_interface.png"))
        
        # Screenshot Task 5: Prediction & Demonstration Table
        print("Submitting review for prediction...")
        # Fill review textarea with a test fake review
        test_text = (
            "This product is absolute junk. It broke the first time I used it. "
            "Do not buy! Waste of money and poor customer support."
        )
        page.fill("textarea#review_text", test_text)
        
        # Click Predict button
        page.click("button[type='submit']")
        page.wait_for_timeout(1500)
        
        print("Capturing Task 5 (Prediction & Demonstration)...")
        # Capture full page showing prediction outcome card and demonstration table
        page.screenshot(path=os.path.join(screenshots_dir, "prediction_demonstration.png"), full_page=True)
        
        # Close browser
        browser.close()
        
    # 3. Terminate Django Server
    print("Stopping Django server...")
    django_process.terminate()
    django_process.wait()
    print("All screenshots generated and saved to 'Screenshots/' folder successfully!")

if __name__ == "__main__":
    generate_all_screenshots()
