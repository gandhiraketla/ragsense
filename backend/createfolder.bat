@echo off

:: Set the project root directory
SET ROOT_DIR=project-root

:: Create the main project folder
mkdir %ROOT_DIR%

:: Create subdirectories
mkdir %ROOT_DIR%\connectors
mkdir %ROOT_DIR%\managers
mkdir %ROOT_DIR%\kafka
mkdir %ROOT_DIR%\config

:: Create empty files
type nul > %ROOT_DIR%\connectors\__init__.py
type nul > %ROOT_DIR%\managers\__init__.py
type nul > %ROOT_DIR%\kafka\__init__.py
type nul > %ROOT_DIR%\config\connector_mapping.json
type nul > %ROOT_DIR%\requirements.txt
type nul > %ROOT_DIR%\main.py

:: Write a sample .gitignore file
echo venv/ > %ROOT_DIR%\.gitignore
echo __pycache__/ >> %ROOT_DIR%\.gitignore

:: Print success message
echo Folder structure created successfully in "%ROOT_DIR%"
pause
cclear