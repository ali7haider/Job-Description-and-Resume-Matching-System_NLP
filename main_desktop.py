import sys
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QMainWindow,
    QFileDialog,
)
import os
import sys
import os
from main_ui import Ui_MainWindow
from PyPDF2 import PdfReader
import docx2txt
from text_processing import preprocess_text
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
from PyQt5.QtGui import QPixmap
import plotly.graph_objects as go
import plotly.io as pio

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Specify a cache directory for Hugging Face models
        cache_dir = ".huggingface_cache"

        # Initialize the DistilBERT tokenizer and model
        print("Initializing DistilBERT tokenizer and model with caching...")
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir)
        # self.model = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir)
        # Set up the user interface from the generated class

        self.modelTrained = Doc2Vec.load('cv_job_maching.model')

        self.setupUi(self)
        from modules.ui_functions import UIFunctions
        self.set_buttons_cursor()
        self.ui=self
        self.setAcceptDrops(True)
        

        self.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        self.btnHome.setStyleSheet(UIFunctions.selectMenu(self.btnHome.styleSheet()))


        # Initialize individual pages
        self.init_pages()

        self.menu_buttons = [
        self.btnHome,  # Replace with your actual button objects
        self.btnSingle,  # Replace with your actual button objects
    ]

        # Assign menu button clicks
        self.btnHome.clicked.connect(self.show_home_page)
        self.btnSingle.clicked.connect(self.show_single_page)

        self.btnNext.clicked.connect(self.handle_next_button_click)
        self.btnNext_2.clicked.connect(self.handle_next_button_click_2)

        
        self.btnUpload.clicked.connect(self.handle_upload_button_click)
        self.btnUpload_2.clicked.connect(self.handle_upload_button_click_2)


        self.btnAnalyze.clicked.connect(self.start_nlp_process)
        self.btnAnalyze_2.clicked.connect(self.start_nlp_process_2)

        self.btnFinish.clicked.connect(self.handle_finish_button_click)
        self.btnFinish_2.clicked.connect(self.handle_finish_button_click_2)


        self.saved_job_description=''
        self.saved_job_description_2=''
        self.default_resume_count=3
        self.num_resumes=0
        self.saved_resumes={}
        self.input_resume=''


    def init_pages(self):
        """Initialize backend logic for each page."""
        # Initialize ConfigSystem logic
        self.stackedWidget.setCurrentIndex(0)


    def show_home_page(self):
        self.handleMenuClick(self.btnHome, 0)
    def show_single_page(self):
        self.handleMenuClick(self.btnSingle, 3)
    
    
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


    def handle_next_button_click_2(self):
        try:
            job_description_text = self.jobDescription_2.toPlainText()

            # Check the length of the job description
            word_count = len(job_description_text.split())
            if word_count < 30 or word_count > 2500:
                QMessageBox.warning(self, "Error", "The job description you entered is too short. Please make sure you paste in a job description that's at least 30 words and a maximum of 2500 words.")
            else:
                # Save the text if conditions are satisfied
                self.saved_job_description_2 = job_description_text
                # Switch to the next stack widget page
                self.stackedWidget.setCurrentIndex(4)  # Assuming page 2 is index 1
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
                self.lbFilesSelected.setText(f"Files Selected: 0")

                QMessageBox.warning(self, "Warning", "No files selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

    def handle_upload_button_click_2(self):
        try:
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly
            file_filter = "PDF Files (*.pdf);;DOCX Files (*.docx);;Text Files (*.txt)"
            
            # Change to getOpenFileName for selecting a single file
            file, _ = QFileDialog.getOpenFileName(self, "Select File", "", file_filter, options=options)
            
            if file:
                # Save or process the selected file
                self.process_selected_file_2(file)
            else:
                QMessageBox.warning(self, "Warning", "No file selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")

    def process_selected_file_2(self, file):
        """Handles processing of the selected file."""   
     
        try:
            # Process the selected file based on its extension
            if file.endswith('.pdf'):
                text = self.extract_text_from_pdf(file)
            elif file.endswith('.docx'):
                text = self.extract_text_from_docx(file)
            elif file.endswith('.txt'):
                text = self.extract_text_from_txt(file)
            else:
                raise ValueError(f"Unsupported file type: {file}")
            self.lbFilesSelected_2.setText(f"Files Selected: Successfully")


            # Store the resume text (for NLP processing or other use)
            self.input_resume = text
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to process file {file}: {str(e)}")
            print(f"Error processing file {file}: {str(e)}")
    def process_selected_files(self, files):
        """Handles processing of the selected files."""
        self.selected_files_count = len(files)
        self.lbFilesSelected.setText(f"Files Selected: {self.selected_files_count}")
        self.saved_resumes={}
        for file_path in files:
            try:
                if file_path.endswith('.pdf'):
                    text = self.extract_text_from_pdf(file_path)
                elif file_path.endswith('.docx'):
                    text = self.extract_text_from_docx(file_path)
                elif file_path.endswith('.txt'):
                    text = self.extract_text_from_txt(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_path}")

                self.saved_resumes[file_path] = text

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to process file {file_path}: {str(e)}")
                print(f"Error processing file {file_path}: {str(e)}")
    def start_nlp_process(self):
        """Starts the NLP process on the selected resumes."""
        if not self.saved_resumes:
            QMessageBox.warning(self, "Warning", "No files selected to analyze.")
            return

        try:
            txtNoResumes = self.txtNoResumes.text().strip()  # Assume txtNoResumes is a QTextEdit or QLineEdit
            self.num_resumes = int(txtNoResumes) if txtNoResumes else self.default_resume_count

            if len(self.saved_resumes) < self.num_resumes:
                QMessageBox.warning(self, "Warning", f"Not enough resumes uploaded. At least {self.num_resumes} resumes are required.")
            else:
                # Proceed with NLP processing
                self.stackedWidget.setCurrentIndex(2)
                self.analyze_resumes_for_nlp()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid number entered for minimum resumes. Please enter a valid integer.")

    def start_nlp_process_2(self):
        """Starts the NLP process on the selected resumes."""
        if not self.input_resume:
            QMessageBox.warning(self, "Warning", "No files selected to analyze.")
            return

        try:
            self.stackedWidget.setCurrentIndex(5)
            self.analyze_resumes_for_nlp_2()
        except ValueError:
            QMessageBox.warning(self, "Error", "Invalid number entered for minimum resumes. Please enter a valid integer.")

    def analyze_resumes_for_nlp(self):
        """Start NLP processing in a background thread."""
        try:
            # Create the NLPWorker thread
            self.nlp_worker = NLPWorker(self.saved_resumes, self.saved_job_description, self.tokenizer, self.model, self.default_resume_count)

            # Connect signals for UI updates
            self.nlp_worker.update_label.connect(self.label_5.setText)
            self.nlp_worker.update_results.connect(self.display_nlp_results)

            # Start the background thread
            self.nlp_worker.start()

            # Change to the page with the processing label
            self.stackedWidget.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while starting NLP processing: {str(e)}")
            print(f"An error occurred: {str(e)}")
    def analyze_resumes_for_nlp_2(self):
            """Start NLP processing in a background thread."""
            try:
                self.single_resume_worker = SingleResumeWorker(self.input_resume, self.saved_job_description_2, self.modelTrained)
                self.single_resume_worker.update_label.connect(self.label_5.setText)
                self.single_resume_worker.update_similarity_result.connect(self.display_similarity_result)

                # Start the thread for single resume
                self.single_resume_worker.start()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while starting NLP processing: {str(e)}")
                print(f"An error occurred: {str(e)}")

    def display_similarity_result(self, similarity_score):
        """Displays the similarity score of a single resume compared to the job description."""
        try:
            self.txtSingleResult.setText(f"Similarity Score: {similarity_score}%")
            self.plot_gauge_in_label(similarity_score)

            print(f"Similarity Score: {similarity_score}%")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while displaying similarity: {str(e)}")
            print(f"An error occurred: {str(e)}")

    def display_nlp_results(self, results):
        """Displays the NLP results (most similar resumes) in the UI."""
        try:
            self.resultsList.clear()  # Clear any existing content

            # Check if there are results to display
            if not results:
                QMessageBox.warning(self, "Warning", "No similar resumes found.")
                return

            for i, (file_path, score) in enumerate(results):
                try:
                    file_name = os.path.basename(file_path)  # Get the file name from the file path
                    # Append the result to the QTextEdit
                    self.resultsList.append(f"{i+1}. {file_name} (Similarity Score: {score:.3f})")
                except Exception as e:
                    # Handle any unexpected errors for this specific resume
                    QMessageBox.warning(self, "Warning", f"Error processing resume: {file_path}")
                    print(f"Error processing file {file_path}: {str(e)}")

            print("Results displayed successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while displaying results: {str(e)}")
            print(f"An error occurred: {str(e)}")
    

    def handle_finish_button_click(self):
        # Change to the first page of the stack widget
        self.stackedWidget.setCurrentIndex(0)

        # Clear the job description text
        self.jobDescription.clear()
    def handle_finish_button_click_2(self):
            # Change to the first page of the stack widget
            self.stackedWidget.setCurrentIndex(3)

            # Clear the job description text
            self.jobDescription_2.clear()

    def plot_gauge_in_label(self, similarity):
        # Create the Plotly gauge figure
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = similarity,
            mode = "gauge+number",
            title = {'text': "Matching percentage (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'steps' : [
                    {'range': [0, 50], 'color': "#FFB6C1"},
                    {'range': [50, 70], 'color': "#FFFFE0"},
                    {'range': [70, 100], 'color': "#90EE90"}
                ],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))

        fig.update_layout(width=600, height=400)  # Adjust the width and height as desired

        # Save the figure as an image
        try:
            image_path = 'gauge_plot.png'  # Set the desired file path and format (e.g., .png, .jpg)
            fig.write_image(image_path)
            print(f"Plot saved successfully as {image_path}")
            pixmap = QPixmap('gauge_plot.png')
            # Display the pixmap in the label (QLabel)
            self.lblGraph.setPixmap(pixmap)
        except Exception as e:
            print(f"Error saving plot: {e}")


    
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
            self.process_selected_files(file_paths)
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


    

from PyQt5.QtCore import QThread, pyqtSignal

class NLPWorker(QThread):
    # Define signals for updating UI
    update_label = pyqtSignal(str)
    update_results = pyqtSignal(list)

    def __init__(self, saved_resumes, job_description, tokenizer, model, default_resume_count):
        super().__init__()
        self.saved_resumes = saved_resumes
        self.job_description = job_description
        self.tokenizer = tokenizer
        self.model = model
        self.default_resume_count = default_resume_count

    def run(self):
        try:
            if not self.saved_resumes:
                self.update_label.emit("No files selected to analyze.")
                return

            if not self.job_description:
                self.update_label.emit("Job description is empty. Please provide a valid job description.")
                return

            preprocessed_job_desc = preprocess_text(self.job_description)

            self.update_label.emit("Starting NLP analysis...")
            job_desc_tokens = self.tokenizer(preprocessed_job_desc, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                job_desc_embedding = self.model(**job_desc_tokens).last_hidden_state.mean(dim=1).numpy()

            self.update_label.emit("Generating embeddings for resumes...")
            preprocessed_resumes = [preprocess_text(resume) for resume in self.saved_resumes.values()]

            resume_tokens = self.tokenizer(preprocessed_resumes, padding=True, truncation=True, return_tensors='pt')

            with torch.no_grad():
                output = self.model(**resume_tokens).last_hidden_state
            resume_embeddings = output.mean(dim=1).numpy()

            self.update_label.emit("Calculating cosine similarity...")
            similarity_scores = cosine_similarity(job_desc_embedding, resume_embeddings)[0]

            # Map scores to file paths
            results = sorted(zip(self.saved_resumes.keys(), similarity_scores), key=lambda x: x[1], reverse=True)
            self.update_label.emit("Displaying Results")

            self.update_results.emit(results[:self.default_resume_count])

        except Exception as e:
            self.update_label.emit(f"An error occurred: {str(e)}")
            print(f"An error occurred: {str(e)}")
class SingleResumeWorker(QThread):
    update_label = pyqtSignal(str)
    update_similarity_result = pyqtSignal(float)

    def __init__(self, input_resume, input_job_desc, model, parent=None):
        super().__init__(parent)
        self.input_resume = input_resume
        self.input_job_desc = input_job_desc
        self.model = model

    def run(self):
        try:
            # Preprocess and infer vectors for both resume and job description
            v1 = self.model.infer_vector(self.input_resume.split())
            v2 = self.model.infer_vector(self.input_job_desc.split())
            
            # Calculate similarity
            similarity = 100 * (np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
            similarity_rounded = round(similarity, 2)

            # Emit the result
            self.update_similarity_result.emit(similarity_rounded)
        
        except Exception as e:
            self.update_label.emit(f"Error: {str(e)}")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
