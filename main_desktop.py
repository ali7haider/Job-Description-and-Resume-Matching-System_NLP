import sys
import os
from PyQt5.QtCore import Qt
from xml.etree import ElementTree as ET
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QMainWindow,
    QFileDialog,
)
from PyQt5.QtWidgets import QApplication, QPushButton, QMessageBox, QMainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow,QPushButton
import os
import sys
#from androguard.core.bytecodes.apk import APK
from PyQt5.QtWidgets import QMainWindow, QPushButton,QStackedWidget
import os
from PyQt5 import uic
from main_ui import Ui_MainWindow
from PyPDF2 import PdfReader
import docx2txt
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Set up the user interface from the generated class
        self.setupUi(self)
        from modules.ui_functions import UIFunctions
        self.set_buttons_cursor()
        self.ui=self
        

        self.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        self.btnHome.setStyleSheet(UIFunctions.selectMenu(self.btnHome.styleSheet()))


        # Initialize individual pages
        self.init_pages()

        self.menu_buttons = [
        self.btnHome,  # Replace with your actual button objects
    ]

        # Assign menu button clicks
        self.btnHome.clicked.connect(self.show_home_page)

        self.btnNext.clicked.connect(self.handle_next_button_click)
        self.btnUpload.clicked.connect(self.handle_upload_button_click)
        self.btnAnalyze.clicked.connect(self.start_nlp_process)


        self.saved_job_description=''


    def init_pages(self):
        """Initialize backend logic for each page."""
        # Initialize ConfigSystem logic
        self.stackedWidget.setCurrentIndex(0)


    def show_home_page(self):
        self.handleMenuClick(self.btnHome, 0)
    
    
    def handle_next_button_click(self):
        try:
            job_description_text = self.jobDescription.toPlainText()

            # Check the length of the job description
            word_count = len(job_description_text.split())
            if word_count < 30 or word_count > 2500:
                QMessageBox.warning(self, "Error", "The job description you entered is too short. Please make sure you paste in a job description that's at least 30 words and a maximum of 2500 words.")
            else:
                # Save the text if conditions are satisfied
                self.saved_job_description = job_description_text
                # Switch to the next stack widget page
                self.stackedWidget.setCurrentIndex(1)  # Assuming page 2 is index 1
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")


    def handle_upload_button_click(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_filter = "PDF Files (*.pdf);;DOCX Files (*.docx);;Text Files (*.txt)"
            
            files, _ = QFileDialog.getOpenFileNames(self, "Select Files", "", file_filter, options=options)
            
            if files:
                # Save or process the selected files
                self.process_selected_files(files)
            else:
                QMessageBox.warning(self, "Warning", "No files selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")


    

    def process_selected_files(self, files):
        """Handles processing of the selected files."""
        self.selected_files_count = len(files)
        self.lbFilesSelected.setText(f"Files Selected: {self.selected_files_count}")

        for file_path in files:
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                text = self.extract_text_from_docx(file_path)
            elif file_path.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
            else:
                print(f"Unsupported file type: {file_path}")
                continue

            self.saved_resumes[file_path] = text
    def start_nlp_process(self):
        """Starts the NLP process on the selected resumes."""
        if not self.saved_resumes:
            QMessageBox.warning(self, "Warning", "No files selected to analyze.")
        else:
            # Placeholder for NLP processing logic
            self.analyze_resumes_for_nlp()

    def analyze_resumes_for_nlp(self):
        """Analyzes the resumes for the most similar ones based on the job description."""
        # Implement the actual NLP logic here
        print("Analyzing resumes for NLP...")

        # Example NLP processing (replace with actual NLP logic)
        results = self.compare_resumes_with_job_description(self.saved_resumes)
        self.display_nlp_results(results)

    def compare_resumes_with_job_description(self, resumes):
        """Compares resumes against the job description and returns the most similar ones."""
        # Placeholder function for comparing resumes to job description
        # Example comparison logic (needs to be replaced with actual NLP techniques)
        results = sorted(resumes.items(), key=lambda x: len(x[1]), reverse=True)  # Sorting by length as a placeholder
        return results[:5]  # Return top 5 similar resumes

    def display_nlp_results(self, results):
        """Displays the NLP results (most similar resumes) in the UI."""
        # Update UI with the most similar resumes
        self.resultsList.clear()
        for i, (file_path, text) in enumerate(results):
            self.resultsList.addItem(f"{i+1}. {file_path} (Length: {len(text)} characters)")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    def dropEvent(self, event):
        """Handle drop event based on the active page with multiple files."""
        if not event.mimeData().hasUrls():
            return event.ignore()

        file_paths = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                file_paths.append(file_path)

        if file_paths:
            self.handle_multiple_files_drop(file_paths)


    def handle_multiple_files_drop(self, file_paths):
        """Handles multiple file drops."""
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(file_path)
                # Do something with the extracted text, e.g., display or store it
            elif file_path.endswith('.docx'):
                text = self.extract_text_from_docx(file_path)
                # Do something with the extracted text
            elif file_path.endswith('.txt'):
                text = self.extract_text_from_txt(file_path)
                # Do something with the extracted text
            else:
                print(f"Unsupported file type: {file_path}")

    def process_directory(self, directory):
        """Handles directory processing to locate .cs, .pdf, .txt, and .docx files."""
        files_found = {'pdf': None, 'docx': None, 'txt': None}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.pdf'):
                    files_found['pdf'] = os.path.join(root, file)
                elif file.endswith('.docx'):
                    files_found['docx'] = os.path.join(root, file)
                elif file.endswith('.txt'):
                    files_found['txt'] = os.path.join(root, file)
                    
            if all(files_found.values()):
                break

        # Process each file if found
        for file_type, file_path in files_found.items():
            if file_path:
                if file_type == 'pdf':
                    text = self.extract_text_from_pdf(file_path)
                    # Do something with the extracted text
                elif file_type == 'docx':
                    text = self.extract_text_from_docx(file_path)
                    # Do something with the extracted text
                elif file_type == 'txt':
                    text = self.extract_text_from_txt(file_path)
                    # Do something with the extracted text
    def extract_text_from_docx(self,file_path):
        return docx2txt.process(file_path)

    def extract_text_from_pdf(self,file_path):
        text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text
    def extract_text_from_txt(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    
    def handleMenuClick(self, button, page_index):
        """
        Handles menu button clicks to update styles and switch pages.
        """
        from modules.ui_functions import UIFunctions

        # Deselect all buttons
        for btn in self.menu_buttons:
            btn.setStyleSheet(UIFunctions.deselectMenu(btn.styleSheet()))

        # Select the clicked button
        button.setStyleSheet(UIFunctions.selectMenu(button.styleSheet()))

        # Switch to the selected page
        self.stackedWidget.setCurrentIndex(page_index)


    def set_buttons_cursor(self):
        """Set the pointer cursor for all buttons in the UI."""
        buttons = self.findChildren(QPushButton)  # Find all QPushButton objects
        for button in buttons:
            button.setCursor(Qt.PointingHandCursor)


    



if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
