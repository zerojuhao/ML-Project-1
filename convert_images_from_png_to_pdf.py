from PIL import Image

def png_to_pdf(png_image_path, pdf_file_path):
    image = Image.open(png_image_path)
    pdf_image = image.convert('RGB')
    pdf_image.save(pdf_file_path, 'PDF', resolution=100.0)

if __name__ == "__main__":
    png_to_pdf('C:/Users/57021/Desktop/ML-Project-1/images/DTU_LOGO.png', 'C:/Users/57021/Desktop/ML-Project-1/images/DTU_LOGO.pdf')
