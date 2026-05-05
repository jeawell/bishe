# 快速部署说明

这份说明适合别人从 GitHub 下载项目后，在自己的电脑上快速运行本项目。

## 环境要求

- Windows 10/11 64 位
- Python 3.10 或 3.11 64 位
- Git 可选；如果是直接下载 zip，不需要 Git

不建议使用 Python 3.12/3.13 作为首次部署环境，因为 `onnxruntime`、`osam` 等 AI 相关依赖在不同机器上更容易出现 DLL 兼容问题。

## 第一次部署

在项目根目录打开 PowerShell 或 Git Bash，也就是包含 `requirements.txt`、`pyproject.toml`、`labelme/` 的目录。

1. 创建虚拟环境：

```bash
python -m venv .venv
```

2. 激活虚拟环境：

PowerShell：

```bash
.\.venv\Scripts\Activate.ps1
```

Git Bash：

```bash
source .venv/Scripts/activate
```

3. 升级 pip：

```bash
python -m pip install --upgrade pip
```

4. 安装依赖：

```bash
pip install -r requirements.txt
```

5. 以可编辑模式安装当前项目：

```bash
pip install -e .
```

## 启动程序

推荐方式：

```bash
python -m labelme
```

如果已经执行过 `pip install -e .`，也可以直接运行：

```bash
labelme
```

## 常见问题

### 1. PowerShell 不允许激活虚拟环境

如果执行 `.\.venv\Scripts\Activate.ps1` 报执行策略错误，可以先运行：

```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新激活虚拟环境。

### 2. onnxruntime DLL load failed

如果启动时报类似错误：

```text
ImportError: DLL load failed while importing onnxruntime_pybind11_state
```

优先处理方式：

```bash
pip uninstall -y onnxruntime osam
pip install onnxruntime osam
```

如果仍然失败，建议换 Python 3.10 或 3.11 重新创建虚拟环境。

### 3. COCO 导出失败

COCO 导出依赖 `pycocotools`。如果安装失败或导出时报缺少 `pycocotools`，可单独安装：

```bash
pip install pycocotools
```

### 4. 不要提交虚拟环境

`.venv/` 是本机环境目录，不需要上传 GitHub。项目已经在 `.gitignore` 中忽略它。
