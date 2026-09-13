"""用户输出管理模块，负责论文结果的拼接、引用处理和保存。"""

import os
import re
from app.utils.data_recorder import DataRecorder
from app.utils.log_util import logger
from app.schemas.A2A import WriterResponse
import json
import uuid


class UserOutput:
    """管理建模任务的输出结果，处理引用编号、脚注和最终论文拼接。"""

    # 行内引用标记：{[^1]: 引用内容}，冒号可省略（历史提示词与模型实际输出常写成
    # {[^1] 引用内容}，用 :? 两者都收，否则引用会被整体丢弃、参考文献章节为空）
    REFERENCE_PATTERN = r"\{\[\^(\d+)\]\s*:?\s*(.*?)\}"

    def __init__(
        self, work_dir: str, ques_count: int, data_recorder: DataRecorder | None = None
    ):
        self.work_dir = work_dir
        self.res: dict[str, dict] = {
            # "eda": {
            #     "response_content": "",
            #     "footnotes": "",
            # },
            # "ques1": {
            #     "response_content": "",
            #     "footnotes": "",
            # },
        }
        self.data_recorder = data_recorder
        self.cost_time = 0.0
        self.initialized = True
        self.ques_count: int = ques_count
        self.footnotes = {}
        self._init_seq()

    def _init_seq(self):
        # 动态顺序获取拼接res value，正确拼接顺序
        ques_str = [f"ques{i}" for i in range(1, self.ques_count + 1)]

        # 修改：调整章节顺序，确保符合论文结构
        self.seq = [
            "firstPage",  # 标题、摘要、关键词
            "RepeatQues",  # 一、问题重述
            "analysisQues",  # 二、问题分析
            "modelAssumption",  # 三、模型假设
            "symbol",  # 四、符号说明和数据预处理
            "eda",  # 四、数据预处理（EDA部分）
            *ques_str,  # 五、模型的建立与求解（问题1、2...）
            "sensitivity_analysis",  # 六、模型的分析与检验
            "judge",  # 七、模型的评价、改进与推广
        ]

    def set_res(self, key: str, writer_response: WriterResponse):
        """设置指定章节的写作结果。

        Args:
            key: 章节标识（如 eda、ques1）。
            writer_response: 写作手的响应对象。
        """
        self.res[key] = {
            "response_content": writer_response.response_content,
            "footnotes": writer_response.footnotes,
        }

    def get_res(self):
        """获取所有章节的写作结果。"""
        return self.res

    def get_model_build_solve(self) -> str:
        """获取模型求解结果的摘要字符串。"""
        model_build_solve = ",".join(
            f"{key}-{value}"
            for key, value in self.res.items()
            if key.startswith("ques") and key != "ques_count"
        )

        return model_build_solve

    def replace_references_with_uuid(self, text: str) -> str:
        """将文本中的引用标记替换为 UUID，用于去重和排序。

        Args:
            text: 包含引用标记的文本。

        Returns:
            替换引用为 UUID 后的文本。
        """

        def _replace(match: re.Match) -> str:
            # 清理引用内容，去除首尾空白和末尾的点号
            ref_content = match.group(2).strip().rstrip(".")

            # 引用内容重复时复用已有的 UUID，保证同一文献只出现一次
            for uuid_key, footnote_data in self.footnotes.items():
                if footnote_data["content"] == ref_content:
                    return f"[{uuid_key}]"

            new_uuid = str(uuid.uuid4())
            self.footnotes[new_uuid] = {"content": ref_content}
            return f"[{new_uuid}]"

        return re.sub(self.REFERENCE_PATTERN, _replace, text, flags=re.DOTALL)

    def sort_text_with_footnotes(self, replace_res: dict) -> dict:
        """按章节顺序排列文本并将 UUID 替换为连续编号。

        Args:
            replace_res: 已替换 UUID 的结果字典。

        Returns:
            按顺序编号后的结果字典。
        """
        sort_res = {}
        ref_index = 1

        for seq_key in self.seq:
            text = replace_res[seq_key]["response_content"]
            # 找到[uuid]（dict.fromkeys 去重且保持出现顺序：同一篇文献多次引用时
            # 复用同一个编号，不会出现正文编号与文末列表对不上的情况）
            uuid_list = dict.fromkeys(re.findall(r"\[([a-f0-9-]{36})\]", text))
            for uid in uuid_list:
                if self.footnotes[uid].get("number") is None:
                    self.footnotes[uid]["number"] = ref_index
                    ref_index += 1
                # 正文里用 [1] 这种「方括号数字」而不是 [^1]：后者在 pandoc/markdown
                # 里是脚注引用，转 docx 时会变成页脚注释、正文里反而看不到参考文献列表
                text = text.replace(f"[{uid}]", f"[{self.footnotes[uid]['number']}]")
            sort_res[seq_key] = {
                "response_content": text,
            }

        return sort_res

    def append_footnotes_to_text(self, text: str) -> str:
        """在文本末尾追加参考文献列表。

        Args:
            text: 论文正文。

        Returns:
            附带参考文献的完整文本。没有引用时原样返回（避免留下一个空标题）。
        """
        if not self.footnotes:
            logger.warning("全文未提取到任何引用标记，跳过「参考文献」章节")
            return text

        # 将脚注转换为列表并按 number 排序
        sorted_footnotes = sorted(
            self.footnotes.items(), key=lambda x: x[1].get("number") or 0
        )
        entries = [
            f"[{footnote['number']}] {footnote['content']}"
            for _, footnote in sorted_footnotes
        ]
        # 条目之间空行分隔：pandoc 会把相邻行合并成同一段，空行才能保证一条一段
        return text + "\n\n## 参考文献\n\n" + "\n\n".join(entries)

    def get_result_to_save(self) -> str:
        """获取最终拼接的论文全文，包含引用处理和参考文献。"""
        replace_res = {}

        for key, value in self.res.items():
            new_text = self.replace_references_with_uuid(value["response_content"])
            replace_res[key] = {
                "response_content": new_text,
            }

        sort_res = self.sort_text_with_footnotes(replace_res)

        full_res_1 = "\n\n".join(
            [sort_res[key]["response_content"] for key in self.seq]
        )

        full_res = self.append_footnotes_to_text(full_res_1)
        return full_res

    def save_result(self):
        """将结果保存为 res.json 和 res.md 文件。"""
        with open(os.path.join(self.work_dir, "res.json"), "w", encoding="utf-8") as f:
            json.dump(self.res, f, ensure_ascii=False, indent=4)

        res_path = os.path.join(self.work_dir, "res.md")
        with open(res_path, "w", encoding="utf-8") as f:
            f.write(self.get_result_to_save())
