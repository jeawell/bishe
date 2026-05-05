import argparse  # 用于解析命令行参数
import codecs  # 用于处理文件编码，特别是UTF-8
import contextlib  # 提供了一些用于处理上下文的工具，如此处的重定向输出
import os
import os.path as osp
import sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import yaml  # 用于读取和解析YAML格式的配置文件
from loguru import logger  # 一个功能强大的日志记录库
from PyQt5 import QtCore  # PyQt5的核心模块，包含信号与槽机制等
from PyQt5 import QtWidgets  # PyQt5的GUI控件模块

from labelme import __appname__
from labelme import __version__
from labelme.app import MainWindow
from labelme.config import get_config
from labelme.utils import newIcon

# 定义一个类，用于将标准错误流重定向到loguru日志记录器
class _LoggerIO:
    def write(self, message: str):
        # 如果消息去除首尾空格后不为空
        if message := message.strip():
            # 将消息作为调试信息写入日志
            logger.debug(message)

    def flush(self):
        pass

# 设置loguru日志记录器的函数
def _setup_loguru(logger_level: str) -> None:
    try:
        # 尝试移除默认的日志处理器，以便自定义
        logger.remove(handler_id=0)
    except ValueError:
        # 如果默认处理器不存在，会抛出ValueError，直接忽略即可
        pass

    if sys.stderr:
        logger.add(sys.stderr, level=logger_level)

    cache_dir: str
    if os.name == "nt":
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], "labelme")
    else:
        cache_dir = os.path.expanduser("~/.cache/labelme")

    os.makedirs(cache_dir, exist_ok=True)

    log_file = os.path.join(cache_dir, "labelme.log")
    logger.add(
        log_file,
        colorize=True,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-V", action="store_true", help="show version")
    parser.add_argument("--reset-config", action="store_true", help="reset qt config")
    parser.add_argument(
        "--logger-level",
        default="debug",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument("filename", nargs="?", help="image or label filename")
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help="output file or directory (if it ends with .json it is "
        "recognized as file, else as directory)",
    )
    # 设置默认配置文件的路径
    # default_config_file = os.path.join(os.path.expanduser("~"), ".labelmerc")
    here = osp.dirname(osp.abspath(__file__))
    default_config_file = os.path.join(here,"config/.labelmerc")
    parser.add_argument(
        "--config",
        dest="config",
        help="config file or yaml-format string (default: {})".format(
            default_config_file
        ),
        default=default_config_file,
    )
    # --- 以下是用于GUI的配置参数 ---
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="epsilon to find nearest vertex on canvas",
        default=argparse.SUPPRESS,
    )
    #  解析命令行传入的参数
    args = parser.parse_args()
    # 如果用户请求显示版本号
    if args.version:
        print("{0} {1}".format(__appname__, __version__))
        sys.exit(0)  # 打印后直接退出程序
    # 设置日志系统
    _setup_loguru(logger_level=args.logger_level.upper())
    # --- 处理可能来自文件或字符串的参数 ---
    # 处理 --flags 参数
    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]
    # 处理 --labels 参数
    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]
    # 处理 --label_flags 参数
    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)
    # 将命令行参数转换为字典，用于后续的配置合并
    config_from_args = args.__dict__
    # 从字典中弹出非配置项的参数
    config_from_args.pop("version")  # pop等于删除
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    # 调用 get_config 函数，它会合并来自文件和命令行的配置
    config = get_config(config_file_or_yaml, config_from_args)
    # 如果开启了标签验证，但没有提供标签列表，则报错退出
    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: true in the config file "
            "(ex. ~/.labelmerc)."
        )
        sys.exit(1)
    # 解析输出路径
    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output
    # 初始化并启动Qt应用程序
    # 加载国际化翻译文件
    translator = QtCore.QTranslator()
    translator.load(
        QtCore.QLocale.system().name(),
        osp.dirname(osp.abspath(__file__)) + "/translate",
    )
    # 创建Qt应用实例
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("newicon"))
    app.installTranslator(translator)
    # 创建主窗口实例，并传入所有配置
    win = MainWindow(
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if reset_config:
        logger.info("Resetting Qt config: %s" % win.settings.fileName())
        win.settings.clear()
        sys.exit(0)

    with logger.catch(), contextlib.redirect_stderr(new_target=_LoggerIO()):  # type: ignore[type-var]
        win.show()
        win.raise_()
        sys.exit(app.exec_())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
