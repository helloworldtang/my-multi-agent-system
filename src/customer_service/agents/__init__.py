"""业务 Agent：FAQ / Order / Complaint，均基于 ``core/agent`` 的薄封装。"""

from customer_service.agents.complaint import make_complaint_agent
from customer_service.agents.faq import make_faq_agent
from customer_service.agents.order import make_order_agent

__all__ = ["make_complaint_agent", "make_faq_agent", "make_order_agent"]
