import os
from fpdf import FPDF


class PDFWithImages(FPDF):
    def add_image_with_caption(self, image_path, caption=None):
        """添加图片和可选的标题"""
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return

        # 获取图片尺寸
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # 计算缩放比例，使图片宽度不超过页面宽度的80%
        max_width = self.w * 0.8
        scale = min(1.0, max_width / img_width)

        # 计算图片位置使其居中
        x = (self.w - img_width * scale) / 2

        # 如果当前页面空间不足，添加新页面
        if self.get_y() + img_height * scale > self.h - 30:
            self.add_page()

        # 添加图片
        self.image(image_path, x=x, y=self.get_y(),
                   w=img_width * scale, h=img_height * scale)

        # 更新Y坐标
        self.set_y(self.get_y() + img_height * scale + 10)

        # 添加标题（如果有）
        if caption:
            self.multi_cell(0, 10, caption, align='C')
            self.ln(10)


def create_pdf_with_fpdf(image_paths, text, output_filename="output.pdf"):
    """
    使用fpdf将图片和文字组合成PDF

    参数:
    image_paths: 图片路径列表
    text: 要添加的文字
    output_filename: 输出PDF文件名
    """
    pdf = PDFWithImages()

    # 添加页面
    pdf.add_page()

    # 设置字体（支持中文需要中文字体文件）
    pdf.set_font("Helvetica", size=20)

    # 添加文字
    pdf.multi_cell(0, 10, text)
    pdf.ln(10)

    # 添加图片
    for i, img_path in enumerate(image_paths):
        # caption = f"picture {i + 1}"  # 可选的图片标题
        pdf.add_image_with_caption(img_path)

    # 保存PDF
    pdf.output(output_filename)
    print(f"PDF已生成: {output_filename}")