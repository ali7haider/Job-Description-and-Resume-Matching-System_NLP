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
        self.ui=self
        self.setAcceptDrops(True)
        self.set_buttons_cursor()
        

        self.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        self.btnHome.setStyleSheet(UIFunctions.selectMenu(self.btnHome.styleSheet()))

        # Access the QStackedWidget
        self.stacked_widget = self.findChild(QStackedWidget, "stackedWidget")  # Match the object name in Qt Designer
        # Initialize individual pages
        self.init_pages()

        self.menu_buttons = [
        self.btnHome,  # Replace with your actual button objects
    ]

        # Assign menu button clicks
        self.btnHome.clicked.connect(self.show_home_page)


    def init_pages(self):
        """Initialize backend logic for each page."""
        # Initialize ConfigSystem logic
        self.stackedWidget.setCurrentIndex(0)


    def show_home_page(self):
        self.handleMenuClick(self.btnHome, 0)

    

    

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
