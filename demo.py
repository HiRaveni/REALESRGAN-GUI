from tkinterdnd2 import DND_FILES, TkinterDnD  # pip install tkinterdnd2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import img2tensor
import logging
import time
import concurrent.futures
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import datetime

# 日志设置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配色方案
COLORS = {
    "background": "#E0E0E0",
    "panel": "#FAFAFA",
    "text": "#424242",
    "primary": "#757575",
    "secondary": "#9E9E9E",
    "accent": "#BDBDBD"
}


class ImageSuperResolutionApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("基于深度学习的图像超分辨率工具")
        self.geometry("1400x800")
        self.configure(bg=COLORS["background"])
        self.create_menu()
        
        self.original_image = None
        self.processed_image = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scale = tk.IntVar(value=4)
        self.current_algorithm = tk.StringVar(value="RealESRGAN")
        self.real_esrgan_tile = tk.IntVar(value=0)
        self.real_esrgan_tile_pad = tk.IntVar(value=10)
        self.real_esrgan_pre_pad = tk.IntVar(value=0)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.history = []
        
        # 记录显示图片的缩放倍率
        self.original_zoom = 1.0
        self.processed_zoom = 1.0
        
        self.init_models()
        self.setup_ui()
        self.bind_wheel_events()

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="上传图片", command=self.upload_image)
        filemenu.add_command(label="加载自定义模型", command=self.load_custom_model)
        filemenu.add_separator()
        filemenu.add_command(label="退出", command=self.quit)
        menubar.add_cascade(label="文件", menu=filemenu)

        history_menu = tk.Menu(menubar, tearoff=0)
        history_menu.add_command(label="查看历史记录", command=self.show_history)
        menubar.add_cascade(label="历史记录", menu=history_menu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="关于", command=lambda: messagebox.showinfo("关于", "张靖杭制作\n版本 0.1"))
        menubar.add_cascade(label="帮助", menu=helpmenu)
        self.config(menu=menubar)

    def init_models(self):
        try:
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            model_path = load_file_from_url(url=model_url, model_dir="weights", progress=True)
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                                 num_block=23, num_grow_ch=32, scale=4)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict["params_ema"])
            self.model.eval().to(self.device)
            if self.device == "cuda":
                self.model.half()
            logging.info("RealESRGAN 模型初始化成功。")
        except Exception as e:
            logging.error(f"模型初始化失败: {e}")
            messagebox.showerror("模型初始化失败", f"初始化模型时出错: {e}")

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        main_frame = tk.Frame(self, bg=COLORS["background"])
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # 左侧图片展示区域
        image_frame = tk.Frame(main_frame, bg=COLORS["panel"])
        image_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        
        # 原图 Canvas 区域
        self.original_frame, self.original_canvas = self.create_scrolled_canvas(image_frame)
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        # 添加拖拽支持和 Canvas 尺寸变化事件绑定
        self.original_canvas.drop_target_register(DND_FILES)
        self.original_canvas.dnd_bind("<<Drop>>", self.drop_image)
        self.original_canvas.bind("<Configure>", self.on_original_canvas_configure)
        
        # 处理后图 Canvas 区域
        self.processed_frame, self.processed_canvas = self.create_scrolled_canvas(image_frame)
        self.processed_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # 控制面板区
        control_frame = tk.LabelFrame(main_frame, text="控制面板", bg=COLORS["background"],
                                      fg=COLORS["text"], font=("Arial", 12, "bold"))
        control_frame.grid(row=0, column=1, padx=20, pady=20, sticky="n")
        ttk.Label(control_frame, text="选择算法:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        algorithm_dropdown = ttk.Combobox(control_frame, textvariable=self.current_algorithm,
                                          values=["RealESRGAN", "Bicubic"], state="readonly", width=15)
        algorithm_dropdown.grid(row=0, column=1, padx=10, pady=5)
        
        def validate_scale(P):
            return P.isdigit() and int(P) >= 1
        vcmd = (self.register(validate_scale), '%P')
        ttk.Label(control_frame, text="Bicubic 缩放倍数:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=1, column=0, padx=10, pady=5, sticky="w")
        scale_entry = ttk.Entry(control_frame, textvariable=self.scale, width=10,
                                validate="key", validatecommand=vcmd)
        scale_entry.grid(row=1, column=1, padx=10, pady=5)
        
        def validate_non_negative(P):
            return P.isdigit() or P == ""
        vcmd_non_negative = (self.register(validate_non_negative), '%P')
        ttk.Label(control_frame, text="Tile:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=2, column=0, padx=10, pady=5, sticky="w")
        tile_entry = ttk.Entry(control_frame, textvariable=self.real_esrgan_tile,
                               width=10, validate="key", validatecommand=vcmd_non_negative)
        tile_entry.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(control_frame, text="Tile Pad:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=3, column=0, padx=10, pady=5, sticky="w")
        tile_pad_entry = ttk.Entry(control_frame, textvariable=self.real_esrgan_tile_pad,
                                   width=10, validate="key", validatecommand=vcmd_non_negative)
        tile_pad_entry.grid(row=3, column=1, padx=10, pady=5)
        ttk.Label(control_frame, text="Pre Pad:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=4, column=0, padx=10, pady=5, sticky="w")
        pre_pad_entry = ttk.Entry(control_frame, textvariable=self.real_esrgan_pre_pad,
                                  width=10, validate="key", validatecommand=vcmd_non_negative)
        pre_pad_entry.grid(row=4, column=1, padx=10, pady=5)
        
        self.upload_button = ttk.Button(control_frame, text="上传图片", command=self.upload_image)
        self.upload_button.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        self.process_button = ttk.Button(control_frame, text="超分辨率处理", command=self.process_image,
                                         state=tk.DISABLED)
        self.process_button.grid(row=5, column=1, padx=10, pady=10, sticky="ew")
        self.compare_button = ttk.Button(control_frame, text="效果对比", command=self.compare_images,
                                         state=tk.DISABLED)
        self.compare_button.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
        self.save_button = ttk.Button(control_frame, text="保存结果", command=self.save_processed_image,
                                      state=tk.DISABLED)
        self.save_button.grid(row=6, column=1, padx=10, pady=10, sticky="ew")
        
        # 状态显示区域
        status_frame = tk.Frame(main_frame, bg=COLORS["background"])
        status_frame.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ttk.Label(status_frame, text="进度:", background=COLORS["background"],
                  foreground=COLORS["text"]).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.progress_bar["value"] = 0
        self.progress_bar["maximum"] = 100
        self.status_label = ttk.Label(status_frame, text="等待操作...", background=COLORS["background"],
                                      foreground=COLORS["text"])
        self.status_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.time_label = ttk.Label(status_frame, text="处理时间: -", background=COLORS["background"],
                                    foreground=COLORS["text"])
        self.time_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.psnr_label = ttk.Label(status_frame, text="PSNR: -", background=COLORS["background"],
                                    foreground=COLORS["text"])
        self.psnr_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.ssim_label = ttk.Label(status_frame, text="SSIM: -", background=COLORS["background"],
                                    foreground=COLORS["text"])
        self.ssim_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
    
    def create_scrolled_canvas(self, parent):
        frame = tk.Frame(parent, bg=COLORS["panel"])
        canvas = tk.Canvas(frame, bg=COLORS["panel"], bd=0, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        canvas.bind("<ButtonPress-1>", self.start_drag)
        canvas.bind("<B1-Motion>", self.drag)
        return frame, canvas
    
    def bind_wheel_events(self):
        self.original_canvas.bind("<MouseWheel>", self.on_zoom)
        self.processed_canvas.bind("<MouseWheel>", self.on_zoom)
    
    def get_initial_zoom(self, img, canvas):
        # 尝试获取canvas尺寸，若未渲染则使用默认值
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 400, 400
        # 计算缩放比例，确保整图显示并不超过画布
        ratio = min(canvas_width / img.width, canvas_height / img.height, 1.0)
        return ratio
    
    def on_original_canvas_configure(self, event):
        # 当原图Canvas尺寸改变时，重新计算缩放比例并更新显示
        if self.original_image:
            new_zoom = self.get_initial_zoom(self.original_image, self.original_canvas)
            if abs(new_zoom - self.original_zoom) > 0.01:
                self.original_zoom = new_zoom
                self.display_image(self.original_image, self.original_canvas, zoom_factor=self.original_zoom)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.load_image(file_path)
    
    def drop_image(self, event):
        file_path = event.data.split()[0]
        file_path = file_path.strip('{}')
        self.load_image(file_path)
    
    def load_image(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            try:
                img = Image.open(file_path)
                img = img.convert("RGB")
            except Exception as e:
                logging.warning(f"PIL加载失败，尝试cv2: {e}")
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("cv2无法读取图像")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            self.original_image = img
            # 计算初始缩放比例确保图片完整显示在原图Canvas上
            self.original_zoom = self.get_initial_zoom(img, self.original_canvas)
            self.display_image(self.original_image, self.original_canvas, zoom_factor=self.original_zoom)
            self.process_button.config(state=tk.NORMAL)
            self.compare_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.status_label.config(text="图片加载成功！")
        except Exception as e:
            logging.error(f"加载图片失败: {e}")
            messagebox.showerror("错误", f"无法加载图片: {e}")
    
    def display_image(self, img, canvas, zoom_factor=1.0):
        try:
            canvas.delete("all")
            width, height = img.size
            new_width = int(width * zoom_factor)
            new_height = int(height * zoom_factor)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img_resized)
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            canvas.config(scrollregion=(0, 0, new_width, new_height))
            canvas.photo = photo
        except Exception as e:
            logging.error(f"显示图片失败: {e}")
            messagebox.showerror("错误", f"无法显示图片: {e}")
    
    def start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_canvas = event.widget
    
    def drag(self, event):
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self._drag_canvas.scan_dragto(-dx, -dy, gain=1)
        self._drag_start_x = event.x
        self._drag_start_y = event.y
    
    def on_zoom(self, event):
        zoom_step = 0.1
        factor = (1 + zoom_step) if event.delta > 0 else (1 - zoom_step)
        if event.widget == self.original_canvas and self.original_image:
            self.original_zoom *= factor
            self.display_image(self.original_image, self.original_canvas, zoom_factor=self.original_zoom)
        elif event.widget == self.processed_canvas and self.processed_image:
            self.processed_zoom *= factor
            self.display_image(self.processed_image, self.processed_canvas, zoom_factor=self.processed_zoom)
    
    def process_image(self):
        if not self.original_image:
            messagebox.showerror("错误", "请先加载图片。")
            return
        self.status_label.config(text="处理中...")
        start_time = time.time()
        self.thread_pool.submit(self.process_image_thread, start_time)
    
    def process_image_thread(self, start_time):
        try:
            img = self.rgb_to_bgr(self.original_image)
            algorithm = self.current_algorithm.get()
            output = self.process_single_image(img, algorithm)
            if output is not None:
                proc_img = self.bgr_to_rgb(output)
                self.processed_image = proc_img
                self.processed_zoom = 1.0
                self.after(0, self.display_image, self.processed_image, self.processed_canvas, 1.0)
                original_np = np.array(self.original_image)
                processed_np = np.array(proc_img)
                if original_np.shape != processed_np.shape:
                    original_for_metric = cv2.resize(original_np, (processed_np.shape[1],
                                                                   processed_np.shape[0]),
                                                    interpolation=cv2.INTER_CUBIC)
                else:
                    original_for_metric = original_np
                psnr_val = psnr(original_for_metric, processed_np)
                ssim_val = ssim(original_for_metric, processed_np, channel_axis=2)
                processing_time = time.time() - start_time
                self.after(0, self.update_quality_metrics, psnr_val, ssim_val)
                self.after(0, self.update_time_label, processing_time)
                self.after(0, self.update_ui_state, True, "处理完成！")
                history_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "algorithm": algorithm,
                    "scale": self.scale.get(),
                    "tile": self.real_esrgan_tile.get(),
                    "tile_pad": self.real_esrgan_tile_pad.get(),
                    "pre_pad": self.real_esrgan_pre_pad.get(),
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "processing_time": processing_time,
                    "original_preview": self.original_image.copy(),
                    "processed_preview": proc_img.copy()
                }
                self.history.append(history_entry)
                self.after(0, self.compare_button.config, {"state": tk.NORMAL})
                self.after(0, self.save_button.config, {"state": tk.NORMAL})
            else:
                self.after(0, self.update_ui_state, True, "处理失败")
        except Exception as e:
            logging.error(f"处理图片错误: {e}")
            self.after(0, messagebox.showerror, "错误", f"处理图片时出错: {e}")
    
    def process_single_image(self, img, algorithm):
        try:
            funcs = {
                "RealESRGAN": self.process_with_real_esrgan,
                "Bicubic": self.process_with_bicubic
            }
            if algorithm in funcs:
                output = funcs[algorithm](img)
                if output is None:
                    raise ValueError("算法处理返回 None")
                return output
            else:
                raise ValueError("不支持的算法")
        except Exception as e:
            logging.error(f"处理失败: {e}")
            messagebox.showerror("错误", f"处理失败: {e}")
            return None
    
    def save_processed_image(self):
        if not self.processed_image:
            messagebox.showerror("错误", "请先处理图片。")
            return
        folder_path = filedialog.askdirectory()
        if folder_path:
            try:
                file_path = os.path.join(folder_path, "processed_image.png")
                self.processed_image.save(file_path)
                messagebox.showinfo("保存成功", "处理结果已成功保存。")
                logging.info("图片保存成功。")
            except Exception as e:
                logging.error(f"保存图片失败: {e}")
                messagebox.showerror("保存失败", f"保存图片时出错: {e}")
    
    def compare_images(self):
        if not self.original_image or not self.processed_image:
            messagebox.showerror("错误", "请先加载并处理图片。")
            return
        compare_window = tk.Toplevel(self)
        compare_window.title("效果对比")
        compare_window.configure(bg=COLORS["background"])
        
        # 调整原图与处理后图尺寸统一为最大400x400，并取得共同尺寸
        orig = self.original_image.copy()
        proc = self.processed_image.copy()
        orig.thumbnail((400, 400))
        proc.thumbnail((400, 400))
        
        # 取二者最小宽高，调整为相同尺寸
        common_width = min(orig.width, proc.width)
        common_height = min(orig.height, proc.height)
        orig_resized = orig.resize((common_width, common_height), Image.LANCZOS)
        proc_resized = proc.resize((common_width, common_height), Image.LANCZOS)
        
        # 创建图片对象（原图及处理后图保持原尺寸展示）
        orig_photo = ImageTk.PhotoImage(orig)
        proc_photo = ImageTk.PhotoImage(proc)
        
        frame = tk.Frame(compare_window, bg=COLORS["background"])
        frame.pack(padx=10, pady=10)
        
        tk.Label(frame, text="原始图片", bg=COLORS["background"], fg=COLORS["text"]).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(frame, text="处理后图片", bg=COLORS["background"], fg=COLORS["text"]).grid(row=0, column=1, padx=5, pady=5)
        tk.Label(frame, text="混合预览", bg=COLORS["background"], fg=COLORS["text"]).grid(row=0, column=2, padx=5, pady=5)
        
        label_orig = tk.Label(frame, image=orig_photo, bg=COLORS["background"])
        label_orig.image = orig_photo
        label_orig.grid(row=1, column=0, padx=5, pady=5)
        
        label_proc = tk.Label(frame, image=proc_photo, bg=COLORS["background"])
        label_proc.image = proc_photo
        label_proc.grid(row=1, column=1, padx=5, pady=5)
        
        # 初始50%混合图片：使用调整后尺寸的图片进行 blend
        blend_img = Image.blend(orig_resized, proc_resized, alpha=0.5)
        blend_photo = ImageTk.PhotoImage(blend_img)
        label_blend = tk.Label(frame, image=blend_photo, bg=COLORS["background"])
        label_blend.image = blend_photo
        label_blend.grid(row=1, column=2, padx=5, pady=5)
        
        # 节流处理混合预览更新
        blend_job = [None]
        def update_blend(val):
            if blend_job[0] is not None:
                frame.after_cancel(blend_job[0])
            blend_job[0] = frame.after(50, do_blend, val)
        def do_blend(val):
            try:
                alpha = float(val) / 100.0
                new_blend = Image.blend(orig_resized, proc_resized, alpha=alpha)
                new_blend_photo = ImageTk.PhotoImage(new_blend)
                label_blend.configure(image=new_blend_photo)
                label_blend.image = new_blend_photo
            except Exception as e:
                logging.error(f"混合预览更新失败: {e}")
        
        blend_scale = tk.Scale(frame, from_=0, to=100, orient="horizontal", label="混合比例 (%)",
                               command=update_blend, bg=COLORS["background"], fg=COLORS["text"])
        blend_scale.set(50)
        blend_scale.grid(row=2, column=0, columnspan=3, pady=10)
    
    def update_progress(self, progress):
        self.progress_bar["value"] = progress
        self.update_idletasks()
    
    def update_time_label(self, t):
        self.time_label.config(text=f"处理时间: {t:.2f}秒")
    
    def update_quality_metrics(self, psnr_val, ssim_val):
        self.psnr_label.config(text=f"PSNR: {psnr_val:.2f}")
        self.ssim_label.config(text=f"SSIM: {ssim_val:.2f}")
    
    def process_with_real_esrgan(self, img):
        tile = self.real_esrgan_tile.get()
        if tile < 0:
            messagebox.showerror("参数错误", "Tile 大小不能为负数！")
            return None
        tile_pad = self.real_esrgan_tile_pad.get()
        pre_pad = self.real_esrgan_pre_pad.get()
        img_tensor = img2tensor(img.astype(np.float32) / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            img_tensor = img_tensor.half()
        try:
            with torch.no_grad():
                if pre_pad > 0:
                    img_tensor = torch.nn.functional.pad(img_tensor, (pre_pad, pre_pad, pre_pad, pre_pad), mode='reflect')
                if tile > 0:
                    from basicsr.utils.realesrgan_utils import tile_process
                    output = tile_process(self.model, img_tensor, tile, tile_pad)
                else:
                    output = self.model(img_tensor)
                if pre_pad > 0:
                    output = output[:, :, pre_pad:-pre_pad, pre_pad:-pre_pad]
                output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2,1,0], :, :], (1,2,0))
                output = (output * 255.0).round()
            return output.astype(np.uint8)
        except Exception as e:
            logging.error(f"RealESRGAN处理失败: {e}")
            messagebox.showerror("RealESRGAN错误", f"错误信息: {e}")
            return None
    
    def process_with_bicubic(self, img):
        try:
            output = cv2.resize(img, (img.shape[1] * self.scale.get(), img.shape[0] * self.scale.get()),
                                interpolation=cv2.INTER_CUBIC)
            return output
        except Exception as e:
            logging.error(f"Bicubic插值失败: {e}")
            messagebox.showerror("Bicubic错误", f"错误信息: {e}")
            return None
    
    def load_custom_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PTH文件", "*.pth")])
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict["params_ema"])
                logging.info("自定义模型加载成功。")
            except Exception as e:
                logging.error(f"加载自定义模型失败: {e}")
                messagebox.showerror("错误", f"模型加载失败: {e}")
    
    def update_ui_state(self, enabled=True, status="等待操作..."):
        self.process_button.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self.compare_button.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self.save_button.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self.status_label.config(text=status)
    
    def bgr_to_rgb(self, img):
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    def rgb_to_bgr(self, img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    def show_history(self):
        history_window = tk.Toplevel(self)
        history_window.title("历史记录")
        history_window.configure(bg=COLORS["background"])
        for i, entry in enumerate(self.history):
            frame = tk.Frame(history_window, bg=COLORS["background"])
            frame.pack(pady=10, fill="both", expand=True)
            tk.Label(frame, text=f"历史记录 {i+1}  时间: {entry['timestamp']}",
                     bg=COLORS["background"], fg=COLORS["text"], font=("Arial", 12, "bold")).pack(anchor="w")
            tk.Label(frame, text=f"算法: {entry['algorithm']}", bg=COLORS["background"],
                     fg=COLORS["text"]).pack(anchor="w")
            if entry["algorithm"] == "Bicubic":
                tk.Label(frame, text=f"缩放倍数: {entry['scale']}", bg=COLORS["background"],
                         fg=COLORS["text"]).pack(anchor="w")
            else:
                tk.Label(frame, text=f"Tile: {entry['tile']}", bg=COLORS["background"],
                         fg=COLORS["text"]).pack(anchor="w")
                tk.Label(frame, text=f"Tile Pad: {entry['tile_pad']}", bg=COLORS["background"],
                         fg=COLORS["text"]).pack(anchor="w")
                tk.Label(frame, text=f"Pre Pad: {entry['pre_pad']}", bg=COLORS["background"],
                         fg=COLORS["text"]).pack(anchor="w")
            tk.Label(frame, text=f"PSNR: {entry['psnr']:.2f}  SSIM: {entry['ssim']:.2f}  耗时: {entry['processing_time']:.2f}s",
                     bg=COLORS["background"], fg=COLORS["text"]).pack(anchor="w")
            sub_frame = tk.Frame(frame, bg=COLORS["background"])
            sub_frame.pack(pady=5)
            orig_preview = entry["original_preview"].copy()
            orig_preview.thumbnail((150, 150))
            orig_photo = ImageTk.PhotoImage(orig_preview)
            lbl_orig = tk.Label(sub_frame, image=orig_photo, bg=COLORS["background"])
            lbl_orig.image = orig_photo
            lbl_orig.pack(side="left", padx=5)
            proc_preview = entry["processed_preview"].copy()
            proc_preview.thumbnail((150, 150))
            proc_photo = ImageTk.PhotoImage(proc_preview)
            lbl_proc = tk.Label(sub_frame, image=proc_photo, bg=COLORS["background"])
            lbl_proc.image = proc_photo
            lbl_proc.pack(side="left", padx=5)
    
    def quit(self):
        self.thread_pool.shutdown()
        super().quit()


if __name__ == "__main__":
    app = ImageSuperResolutionApp()
    app.mainloop()