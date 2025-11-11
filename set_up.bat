@echo off
rem Use UTF-8 code page so Unicode box-drawing/emoji render correctly in modern terminals
chcp 65001 >nul
title Natural Food Corpus Search Engine - Setup
mode con: cols=100 lines=30
color 0A

setlocal enabledelayedexpansion

:: Parse arguments
set NO_RUN=0
set SKIP_EVAL=0
if "%1"=="--no-run" set NO_RUN=1
if "%1"=="--skip-eval" set SKIP_EVAL=1
if "%2"=="--no-run" set NO_RUN=1
if "%2"=="--skip-eval" set SKIP_EVAL=1

:: Header
echo.
echo ══════════════════════════════════════════════════════════════════════════════════════
echo                      🥗 Natural Food Corpus Search Engine 🔍                          
echo                               Automated Setup Script                                  
echo ══════════════════════════════════════════════════════════════════════════════════════
echo.
echo Usage: set_up.bat [--no-run] [--skip-eval]
echo   --no-run    : Setup and build but do NOT launch Streamlit
echo   --skip-eval : Skip evaluation step
echo.

:: Step 1: Virtual Environment
echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 1: Virtual Environment Setup                                                   │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
if not exist venv (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        echo [HELP] Make sure Python 3.10+ is installed and accessible via 'python' command.
        pause
        exit /b !errorlevel!
    )
    echo [SUCCESS] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment 'venv' already exists. Skipping creation.
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    echo [HELP] Check if venv\Scripts\activate.bat exists.
    pause
    exit /b !errorlevel!
)
echo [SUCCESS] Virtual environment activated.
echo.

:: Step 2: Dependencies
echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 2: Installing Dependencies                                                     │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
set REQ_FILE=requirements.txt
if not exist !REQ_FILE! (
    echo [ERROR] requirements.txt not found.
    echo [HELP] Please provide a requirements file and re-run.
    pause
    exit /b 1
)

echo [INFO] Installing Python packages from !REQ_FILE!...
pip install --upgrade pip >nul 2>&1
pip install -r !REQ_FILE!
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies from !REQ_FILE!.
    echo [HELP] Check your internet connection and Python environment.
    pause
    exit /b !errorlevel!
)
echo [SUCCESS] Dependencies installed successfully.

echo [INFO] Installing spaCy English model 'en_core_web_sm'...
python -m spacy download en_core_web_sm >nul 2>&1
if !errorlevel! neq 0 (
    echo [WARN] spaCy download failed. Trying wheel fallback...
    pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install en_core_web_sm automatically.
        echo [HELP] You may need to install it manually: python -m spacy download en_core_web_sm
        pause
    ) else (
        echo [SUCCESS] en_core_web_sm installed via wheel.
    )
) else (
    echo [SUCCESS] en_core_web_sm installed successfully.
)
echo.

:: Step 3: Environment Configuration
echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 3: Environment Configuration                                                   │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
if not exist ENV.json (
    if exist ENV.json.exp (
        echo [INFO] Creating ENV.json from ENV.json.exp template...
        copy /Y ENV.json.exp ENV.json >nul
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to copy ENV.json.exp to ENV.json
            pause
            exit /b !errorlevel!
        )
        echo [SUCCESS] ENV.json created from template.
        echo [INFO] You can customize ENV.json now. Press any key to open in Notepad...
        pause >nul
        start notepad ENV.json
        echo [INFO] After editing, press any key to continue...
        pause >nul
    ) else (
        echo [ERROR] ENV.json not found and ENV.json.exp template not available.
        echo [HELP] Create ENV.json manually with required paths and re-run.
        pause
        exit /b 1
    )
) else (
    echo [INFO] ENV.json found. Using existing configuration.
    echo [INFO] Do you want to edit ENV.json? Press 'e' + Enter to edit, or just Enter to continue:
    set /p user_choice="> "
    if /i "!user_choice!"=="e" (
        start notepad ENV.json
        echo [INFO] After editing, press any key to continue...
        pause >nul
    )
)
echo.

:: Step 4: Build Inverted Index
echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 4: Building Inverted Index ^(TF-IDF^)                                          │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
echo [INFO] Building TF-IDF inverted index from corpus...
echo [INFO] This may take a few minutes depending on corpus size...
python src\build_inverted_index.py
if !errorlevel! neq 0 (
    echo [ERROR] Failed to build inverted index.
    echo [HELP] Check ENV.json paths and ensure corpus file exists.
    pause
    exit /b !errorlevel!
)
echo [SUCCESS] TF-IDF inverted index built successfully.
echo.

:: Step 4b: Optional Semantic Index
echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 4b: Building Semantic Index ^(Optional^)                                       │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
echo [INFO] Do you want to build a semantic search index? ^(y/n^):
set /p build_semantic="> "
if /i "!build_semantic!"=="y" (
    echo [INFO] Select model:
    echo   1. all-MiniLM-L6-v2 ^(fast, 384d, general-purpose^)
    echo   2. multi-qa-mpnet-base-dot-v1 ^(high quality, 768d, Q^&A^)
    echo   3. NeuML/pubmedbert-base-embeddings ^(biomedical, 768d, RECOMMENDED^)
    set /p model_choice="> "
    
    if "!model_choice!"=="1" (
        set MODEL_NAME=all-MiniLM-L6-v2
        set INDEX_PATH=data\semantic\all-MiniLM-L6-v2\index
    ) else if "!model_choice!"=="2" (
        set MODEL_NAME=sentence-transformers/multi-qa-mpnet-base-dot-v1
        set INDEX_PATH=data\semantic\multi-qa-mpnet-base-dot-v1\index
    ) else if "!model_choice!"=="3" (
        set MODEL_NAME=NeuML/pubmedbert-base-embeddings
        set INDEX_PATH=data\semantic\S-PubMedBert-MS-MARCO\index
    ) else (
        set MODEL_NAME=NeuML/pubmedbert-base-embeddings
        set INDEX_PATH=data\semantic\S-PubMedBert-MS-MARCO\index
        echo [WARN] Invalid choice. Using recommended: PubMedBert
    )
    
    echo [INFO] Building semantic index with model: !MODEL_NAME!...
    echo [INFO] This may take 10-30 minutes depending on corpus size and hardware...
    echo [INFO] GPU acceleration will be used if CUDA is available.
    
    python src\build_index.py --corpus_path data\nfcorpus\corpus.jsonl --index_path !INDEX_PATH! --model_name !MODEL_NAME!
    
    if !errorlevel! neq 0 (
        echo [WARN] Semantic index building failed or was interrupted.
        echo [HELP] You can build it later with: python src\build_index.py --model_name !MODEL_NAME!
    ) else (
        echo [SUCCESS] Semantic index built successfully at !INDEX_PATH!.index
    )
) else (
    echo [INFO] Skipping semantic index building.
    echo [INFO] You can build it later with: python src\build_index.py
)
echo.

:: Step 5: Optional Evaluation
if !SKIP_EVAL!==0 (
    echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
    echo │ STEP 5: Evaluation ^(Optional^)                                                     │
    echo └─────────────────────────────────────────────────────────────────────────────────────┘
    echo [INFO] Do you want to run evaluation? ^(y/n^):
    set /p run_eval="> "
    if /i "!run_eval!"=="y" (
        echo [INFO] Enter number of queries to evaluate ^(e.g., 10, 50, 100^):
        set /p n_queries="> "
        if "!n_queries!"=="" set n_queries=10
        echo [INFO] Running evaluation with !n_queries! queries...
        echo [INFO] This may take several minutes...
        
    :: Create a temporary Python script to run evaluation with custom n_queries
    echo import sys > temp_eval.py
    echo sys.path.append^("."^) >> temp_eval.py
    echo from utils.loader import load_env >> temp_eval.py
    echo env = load_env() >> temp_eval.py
    echo env["N_QUERIES"] = "!n_queries!" >> temp_eval.py
    echo import src.evaluator as evaluator >> temp_eval.py
    echo evaluator.ENV = env >> temp_eval.py
    echo evaluator.main^(^) >> temp_eval.py
        
        python temp_eval.py
        if !errorlevel! neq 0 (
            echo [WARN] Evaluation completed with warnings. Check results folder.
        ) else (
            echo [SUCCESS] Evaluation completed successfully.
        )
        del temp_eval.py >nul 2>&1
    ) else (
        echo [INFO] Skipping evaluation.
    )
    echo.
)

:: Step 6: Launch Streamlit
if !NO_RUN!==1 (
    echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
    echo │ SETUP COMPLETE                                                                      │
    echo └─────────────────────────────────────────────────────────────────────────────────────┘
    echo [SUCCESS] Setup completed successfully. Streamlit launch skipped due to --no-run flag.
    echo [INFO] To start the app manually, run: streamlit run app\streamlit_app.py
    echo.
    pause
    exit /b 0
)

echo ┌─────────────────────────────────────────────────────────────────────────────────────┐
echo │ STEP 6: Launching Streamlit Application                                             │
echo └─────────────────────────────────────────────────────────────────────────────────────┘
echo [INFO] Starting Streamlit web interface...
echo [INFO] The app will open in your default browser at http://localhost:8501
echo [INFO] Press Ctrl+C in this window to stop the server.
echo.
timeout /t 3 >nul
streamlit run app\streamlit_app.py
if !errorlevel! neq 0 (
    echo [ERROR] Failed to start Streamlit app.
    echo [HELP] Check if app\streamlit_app.py exists and all dependencies are installed.
    pause
    exit /b !errorlevel!
)

endlocal
