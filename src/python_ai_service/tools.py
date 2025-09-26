from langchain.tools import tool
from typing import List, Dict, Any
import logging

logger = logging.getLogger("python_ai_service.tools")

# Sample orders database
ORDERS_DB = {
    "555-0123": [
        {
            "order_id": "ORD-001",
            "customer_phone": "555-0123",
            "status": "delivered",
            "items": ["Laptop", "Mouse"],
            "total": 1299.99,
            "order_date": "2024-01-15",
        },
        {
            "order_id": "ORD-002",
            "customer_phone": "555-0123",
            "status": "shipped",
            "items": ["Keyboard", "Monitor"],
            "total": 599.99,
            "order_date": "2024-01-20",
        },
    ],
    "555-0456": [
        {
            "order_id": "ORD-003",
            "customer_phone": "555-0456",
            "status": "pending",
            "items": ["Headphones"],
            "total": 199.99,
            "order_date": "2024-01-22",
        }
    ],
    "555-0789": [
        {
            "order_id": "ORD-004",
            "customer_phone": "555-0789",
            "status": "cancelled",
            "items": ["Tablet"],
            "total": 399.99,
            "order_date": "2024-01-18",
        },
        {
            "order_id": "ORD-005",
            "customer_phone": "555-0789",
            "status": "delivered",
            "items": ["Phone Case", "Screen Protector"],
            "total": 49.99,
            "order_date": "2024-01-10",
        },
    ],
}


@tool
def query_orders(customer_phone: str) -> List[Dict[str, Any]]:
    """
    Query orders for a customer by their phone number.

    Args:
        customer_phone: The customer's phone number (e.g., "555-0123")

    Returns:
        List of orders for the customer, or empty list if no orders found
    """
    logger.info(f"Querying orders for customer phone: {customer_phone}")

    orders = ORDERS_DB.get(customer_phone, [])
    logger.info(f"Found {len(orders)} orders for customer {customer_phone}")

    return orders


@tool
def cancel_order(order_id: str) -> Dict[str, Any]:
    """
    Cancel an order by updating its status to 'cancelled'.

    Args:
        order_id: The order ID to cancel (e.g., "ORD-001")

    Returns:
        Dictionary with success status and order details, or error message
    """
    logger.info(f"Attempting to cancel order: {order_id}")

    # Find the order in the database
    for phone, orders in ORDERS_DB.items():
        for order in orders:
            if order["order_id"] == order_id:
                if order["status"] == "cancelled":
                    logger.warning(f"Order {order_id} is already cancelled")
                    return {
                        "success": False,
                        "message": f"Order {order_id} is already cancelled",
                        "order": order,
                    }
                elif order["status"] == "delivered":
                    logger.warning(f"Cannot cancel delivered order {order_id}")
                    return {
                        "success": False,
                        "message": f"Cannot cancel delivered order {order_id}",
                        "order": order,
                    }
                else:
                    # Cancel the order
                    order["status"] = "cancelled"
                    logger.info(f"Successfully cancelled order {order_id}")
                    return {
                        "success": True,
                        "message": f"Order {order_id} has been cancelled",
                        "order": order,
                    }

    logger.warning(f"Order {order_id} not found")
    return {"success": False, "message": f"Order {order_id} not found", "order": None}


# List of available tools for the LLM
AVAILABLE_TOOLS = [query_orders, cancel_order]
