# -*- coding: utf-8 -*-
"""
src/model.py
============

LLM Provider 抽象与 API 调用封装。

设计：
    BaseModel
      ├── OpenAICompatibleModel  （OpenAI / DeepSeek 等 OpenAI 兼容端点）
      │     ├── OpenAIModel
      │     └── DeepSeekModel
      └── MockModel              （仅供测试，绝不进入正式 benchmark 结果）

要点：
  * API Key 一律从环境变量读取，绝不写入代码 / 配置 / 结果 / 日志。
  * 实现 timeout、retry、exponential backoff、rate-limit(429) 与 5xx 处理。
  * 单题失败抛 ModelError，由上层决定是否继续，不会让整批任务直接丢失。
  * 使用 Python 标准库 urllib 发送 OpenAI 兼容的 /chat/completions 请求，
    因此本阶段不需要安装第三方 SDK；若日后要换 openai 包也很容易。
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

# provider -> (环境变量名, 默认 endpoint base_url, 默认模型)
# DeepSeek 现为 V4 系列：deepseek-v4-flash / deepseek-v4-pro。
# 官方 OpenAI 兼容 base_url 为 https://api.deepseek.com。
_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-v4-pro"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
                   "qwen/qwen3.8-max"),
    # 硅基流动：中国站 .cn / 国际站 .com，均 OpenAI 兼容；可用 --base-url 覆盖。
    "siliconflow": ("SILICONFLOW_API_KEY", "https://api.siliconflow.cn/v1",
                    "Qwen/Qwen2.5-7B-Instruct"),
}

CHAT_ENDPOINT = "/chat/completions"


class ModelError(Exception):
    """模型 / API 相关错误。transient=True 表示可重试。"""

    def __init__(self, message, transient=True, retry_after=None):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after


def resolve_api_key(provider):
    """从环境变量读取 API Key，不存在时返回 None。"""
    return os.environ.get(_PROVIDERS[provider][0])


def provider_env_var(provider):
    return _PROVIDERS[provider][0]

class BaseModel:
    """所有模型的基类。子类必须实现 _single_attempt(system, user) -> (content, meta)。"""

    provider = "base"

    def __init__(self, model=None, temperature=0.0, max_tokens=2048,
                 timeout=60, max_retries=3):
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)

    def respond(self, system, user):
        """只返回最终 content 字符串（兼容旧调用/测试）。"""
        content, _meta = self.respond_full(system, user)
        return content

    def respond_full(self, system, user):
        """
        带重试的对外入口：单题最多重试 max_retries 次。
        返回 (content, meta)，meta 含 api_model / finish_reason / usage /
        reasoning_content 等诊断信息（若 API 提供）。
        全部失败则抛 ModelError（由上层决定跳过并继续）。
        """
        last = None
        for attempt in range(1, self.max_retries + 1):
            try:
                content, meta = self._single_attempt(system, user)
                if meta is None:
                    meta = {}
                meta.setdefault("api_model", getattr(self, "model", None))
                return content, meta
            except ModelError as exc:
                last = exc
                if not exc.transient:
                    # 非瞬时错误（如 401/400）不值得重试
                    raise
                if attempt < self.max_retries:
                    # 若服务端给了 Retry-After 就优先遵守，否则指数退避
                    delay = getattr(exc, "retry_after", None)
                    if not delay:
                        delay = self._backoff_delay(attempt)
                    delay = min(float(delay), 60.0)
                    time.sleep(delay)
        raise ModelError(
            "请求在 %d 次尝试后仍失败：%s" % (self.max_retries, last)
        )

    @staticmethod
    def _backoff_delay(attempt, base=1.0, cap=8.0):
        """指数退避 + 随机抖动。"""
        exp = base * (2 ** (attempt - 1))
        jitter = random.uniform(0, 0.5 * base)
        return min(exp + jitter, cap)

    def _single_attempt(self, system, user):
        raise NotImplementedError


class OpenAICompatibleModel(BaseModel):
    """
    通用 OpenAI 兼容 chat 模型。通过 provider 配置 endpoint / key 环境变量。
    """

    def __init__(self, provider, model=None, api_key=None, temperature=0.0,
                 max_tokens=2048, timeout=60, max_retries=3, base_url=None,
                 thinking=None, reasoning_effort=None, enable_thinking=None):
        super().__init__(model=model, temperature=temperature,
                         max_tokens=max_tokens, timeout=timeout,
                         max_retries=max_retries)
        if provider not in _PROVIDERS:
            raise ModelError("不支持的 provider：%s" % provider, transient=False)
        self.provider = provider
        env_name, default_base, default_model = _PROVIDERS[provider]
        if api_key is None:
            api_key = resolve_api_key(provider)
        self.api_key = api_key
        self.env_name = env_name
        self.base_url = (base_url or default_base).rstrip("/")
        if model is None:
            model = default_model
        self.model = model
        # DeepSeek V4 thinking 控制（仅 DeepSeek 生效）
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        # SiliconFlow/Qwen3 的 enable_thinking（仅 siliconflow 生效）
        self.enable_thinking = enable_thinking

    def require_key(self):
        if not self.api_key:
            raise ModelError(
                "%s 未设置。\n请先设置环境变量：\n\nexport %s=\"YOUR_API_KEY\"\n"
                % (self.env_name, self.env_name),
                transient=False,
            )

    def _single_attempt(self, system, user):
        self.require_key()
        url = self.base_url + CHAT_ENDPOINT
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # DeepSeek V4 的 thinking / reasoning_effort（OpenAI 忽略）
        if self.provider == "deepseek":
            if self.thinking in ("enabled", "disabled"):
                payload["thinking"] = {"type": self.thinking}
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        # SiliconFlow / Qwen3 的 enable_thinking（仅 siliconflow 生效）
        if self.provider == "siliconflow" and self.enable_thinking is not None:
            payload["enable_thinking"] = bool(self.enable_thinking)

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            code = exc.code
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            # 402：OpenRouter 在途预算/额度类错误（in_flight_budget_exhausted）
            #     属于可重试；真正的欠费也会在重试耗尽后失败，不会误判成功。
            # 408/425/429/5xx 同样可重试。其它 4xx（如 401/400/404）不重试。
            transient = (code in (402, 408, 425, 429) or code >= 500)
            retry_after = None
            try:
                ra = exc.headers.get("Retry-After") if exc.headers else None
                if ra:
                    retry_after = float(ra)
            except (TypeError, ValueError):
                retry_after = None
            if code == 402 and "in_flight_budget_exhausted" in detail:
                transient = True
            message = "HTTP %d 错误（%s）：%s" % (code, self.model, detail)
            if code == 401:
                key = self.api_key or ""
                masked = "len=%d, tail=...%s" % (len(key), key[-4:])
                message += (
                    "\n→ API Key 被拒绝。请检查："
                    "\n  1) 当前请求端点：%s（中国站应为 https://api.siliconflow.cn/v1，"
                    "国际站应为 https://api.siliconflow.com/v1）"
                    "\n  2) 当前 key：%s（应与站点匹配；注意不要有多余空格/引号）"
                    "\n  3) 在运行 python 的同一个终端里重新 export 最新 key"
                    % (self.base_url, masked)
                )
            raise ModelError(
                message,
                transient=transient,
                retry_after=retry_after,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelError("网络/超时错误：%s" % exc, transient=True)

        try:
            choice = body["choices"][0]
            message = choice["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelError(
                "响应格式异常（缺少 choices/message/content）：%s" % exc,
                transient=True,
            )

        content = content if isinstance(content, str) else str(content or "")
        usage = body.get("usage")
        reasoning_tokens = None
        if isinstance(usage, dict):
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens")
        meta = {
            "api_model": body.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "native_finish_reason": choice.get("native_finish_reason"),
            "usage": usage,
            "reasoning_tokens": reasoning_tokens,
            "reasoning_content": message.get("reasoning_content"),
        }
        return content, meta

class OpenAIModel(OpenAICompatibleModel):
    """OpenAI。"""

    def __init__(self, model=None, **kwargs):
        super().__init__(provider="openai", model=model, **kwargs)


class DeepSeekModel(OpenAICompatibleModel):
    """DeepSeek（OpenAI 兼容端点 https://api.deepseek.com）。默认 V4 Pro。"""

    def __init__(self, model=None, **kwargs):
        super().__init__(provider="deepseek", model=model, **kwargs)


class OpenRouterModel(OpenAICompatibleModel):
    """
    OpenRouter（OpenAI 兼容端点 https://openrouter.ai/api/v1）。

    - Key 从 OPENROUTER_API_KEY 读取。
    - model 原样使用 CLI 传入的 OpenRouter model ID（含 provider 前缀，如
      qwen/qwen3.8-max、google/gemini-3.8-flash），绝不添加/删除前缀。
    - 不注入 DeepSeek 专有的 thinking / reasoning_effort 参数，
      避免对不支持的模型硬编码行为（相关参数仅在 DeepSeek provider 层处理）。
    """

    def __init__(self, model=None, **kwargs):
        super().__init__(provider="openrouter", model=model, **kwargs)


class SiliconFlowModel(OpenAICompatibleModel):
    """
    硅基流动 SiliconFlow（OpenAI 兼容）。

    - Key 从 SILICONFLOW_API_KEY 读取。
    - 默认端点 https://api.siliconflow.cn/v1（国际站可用 --base-url
      https://api.siliconflow.com/v1 覆盖）。
    - model 原样使用 CLI 传入的 SiliconFlow model ID（如
      Qwen/Qwen2.5-7B-Instruct、deepseek-ai/DeepSeek-V3），绝不改写。
    - 不注入 DeepSeek 专有的 thinking / reasoning_effort 参数。
    """

    def __init__(self, model=None, **kwargs):
        super().__init__(provider="siliconflow", model=model, **kwargs)


# provider -> 具体模型类（均复用 OpenAICompatibleModel 的 HTTP/重试逻辑）
_MODEL_CLASSES = {
    "openai": OpenAIModel,
    "deepseek": DeepSeekModel,
    "openrouter": OpenRouterModel,
    "siliconflow": SiliconFlowModel,
}


def build_model(provider, model=None, temperature=0.0, max_tokens=2048,
                timeout=60, max_retries=3, mock_answer=None,
                mock_fail_first=0, thinking=None, reasoning_effort=None,
                base_url=None, enable_thinking=None):
    """
    根据 provider 名称构建模型实例。
      provider == "mock" -> MockModel（只用于测试，不含任何 API Key）。
    其它走 OpenAICompatibleModel（OpenAI / DeepSeek / OpenRouter / SiliconFlow）。
    thinking / reasoning_effort 仅对 DeepSeek 生效；
    enable_thinking 仅对 SiliconFlow 生效；base_url 可覆盖默认端点。
    """
    provider = provider.lower()
    if provider == "mock":
        return MockModel(model=model, temperature=temperature,
                         max_tokens=max_tokens, timeout=timeout,
                         max_retries=max_retries, canned=mock_answer,
                         fail_first=mock_fail_first)
    cls = _MODEL_CLASSES.get(provider, OpenAICompatibleModel)
    kwargs = dict(model=model, temperature=temperature,
                  max_tokens=max_tokens, timeout=timeout,
                  max_retries=max_retries, base_url=base_url,
                  thinking=thinking, reasoning_effort=reasoning_effort,
                  enable_thinking=enable_thinking)
    if cls is OpenAICompatibleModel:
        kwargs["provider"] = provider
    return cls(**kwargs)


class MockModel(BaseModel):
    """
    测试用假模型，不联网、不需要 API Key。
      canned  : 固定回答（str）。若为 None，则根据 user 中是否出现"双选题"，
                分别返回 "CD"（two_choice）或 "B"（single_choice）。
      fail_first: 前 N 次调用抛瞬时错误，用于测试 retry。
    说明：mock 绝不能作为正式 benchmark 结果对外发布。
    """

    provider = "mock"

    def __init__(self, model="mock-chat", canned=None, fail_first=0, **kwargs):
        super().__init__(model=model, **kwargs)
        self.canned = canned
        self.fail_first = int(fail_first)
        self.calls = 0

    def _single_attempt(self, system, user):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise ModelError("mock 模拟瞬时失败（第 %d 次）" % self.calls,
                             transient=True)
        if self.canned is not None:
            content = self.canned
        else:
            content = "CD" if "双选题" in user else "B"
        return content, {"api_model": self.model, "finish_reason": "stop"}
