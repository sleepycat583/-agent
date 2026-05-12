import logging
import os
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_system_prompt(prompt_path: Path) -> str:
    logger.info("Loading system prompt from %s", prompt_path)
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def summarize_meeting(meeting_text: str, model: str = "deepseek-chat") -> str:
    base_dir = Path(__file__).resolve().parent
    prompt_path = base_dir / "prompt.txt"
    system_prompt = load_system_prompt(prompt_path)

    # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    logger.info("Sending meeting text to DeepSeek model: %s", model)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": meeting_text},
            ],
            temperature=0.2,
            timeout=15,
        )
    except APITimeoutError as e:
        logger.exception("DeepSeek API request timed out: %s", e)
        raise
    except APIConnectionError as e:
        logger.exception("DeepSeek API connection failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error during DeepSeek API call: %s", e)
        raise

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty content.")

    logger.info("Received meeting summary successfully")
    return content


if __name__ == "__main__":
    logger.info("Starting meeting summary MVP demo")

    mock_meeting_text = """
    时间：2026-05-12 上午项目例会
    参会人：李明（产品）、王强（研发）、陈婷（测试）、赵敏（运营）

    会议内容：
    1. 产品李明：本周核心目标是上线“企业版权限管理”MVP，必须先支持角色模板和成员批量导入。
    2. 研发王强：后端接口预计 5 月 16 日前完成，前端管理页需要到 5 月 18 日。风险是权限继承逻辑复杂，可能增加 1 天联调时间。
    3. 测试陈婷：建议 5 月 17 日开始冒烟测试，5 月 19 日完成主流程回归。前提是 5 月 16 日能拿到稳定构建版本。
    4. 运营赵敏：需要在 5 月 20 日前准备上线公告、FAQ 和客户通知邮件，依赖产品在 5 月 18 日前提供最终功能清单。
    5. 决议：目标上线时间暂定 5 月 21 日，若联调延期超过 1 天，则顺延到 5 月 23 日。
    """

    try:
        result = summarize_meeting(mock_meeting_text)
        print("\n===== 会议提要输出 =====\n")
        print(result)
    except Exception:
        logger.error("Meeting summary demo failed.")
