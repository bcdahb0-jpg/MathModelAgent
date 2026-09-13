"""UserOutput 引用提取与参考文献拼接的单元测试。"""

import unittest

from app.models.user_output import UserOutput


class TestReferenceExtraction(unittest.TestCase):
    """测试行内引用标记 → UUID → 编号 → 参考文献列表的完整链路。"""

    def setUp(self):
        self.output = UserOutput(work_dir=".", ques_count=1)

    def test_extract_with_colon(self):
        """{[^1]: 引用内容} 形态能被提取（提示词的标准写法）。"""
        text = self.output.replace_references_with_uuid("正文{[^1]: 张三 (2020). 一篇论文}")
        self.assertNotIn("{[^", text)
        self.assertEqual(len(self.output.footnotes), 1)
        self.assertEqual(
            next(iter(self.output.footnotes.values()))["content"], "张三 (2020). 一篇论文"
        )

    def test_extract_without_colon(self):
        """{[^1] 引用内容} 形态也要能提取（模型常漏掉冒号，曾导致参考文献整章为空）。"""
        text = self.output.replace_references_with_uuid("正文{[^1] 李四 (2021). 另一篇论文}")
        self.assertNotIn("{[^", text)
        self.assertEqual(len(self.output.footnotes), 1)
        self.assertEqual(
            next(iter(self.output.footnotes.values()))["content"], "李四 (2021). 另一篇论文"
        )

    def test_duplicated_reference_reuses_uuid(self):
        """同一篇文献重复引用时复用同一个 UUID，最终只占一个编号。"""
        text = self.output.replace_references_with_uuid(
            "A{[^1]: 同一篇} B{[^2] 同一篇} C{[^3]: 另一篇}"
        )
        self.assertEqual(len(self.output.footnotes), 2)
        uuids = [u for u in self.output.footnotes if u in text]
        self.assertEqual(len(uuids), 2)

    def test_references_appended_to_text(self):
        """引用最终要落到文末的「参考文献」列表里，正文用 [n] 编号。

        不能用 [^n]：那是 pandoc/markdown 的脚注语法，转 docx 后会变成页脚注释，
        正文末尾的「参考文献」标题下面反而是空的。
        """
        self.output.res = {
            "firstPage": {"response_content": "摘要{[^1]: 张三 (2020). 论文A}"}
        }
        self.output.seq = ["firstPage"]
        full = self.output.get_result_to_save()
        self.assertIn("## 参考文献", full)
        self.assertIn("[1] 张三 (2020). 论文A", full)
        self.assertIn("摘要[1]", full)
        self.assertNotIn("[^", full)

    def test_no_reference_no_heading(self):
        """没有任何引用时不留一个空的「参考文献」标题。"""
        self.output.res = {"firstPage": {"response_content": "摘要，没有引用"}}
        self.output.seq = ["firstPage"]
        full = self.output.get_result_to_save()
        self.assertNotIn("参考文献", full)


if __name__ == "__main__":
    unittest.main()
