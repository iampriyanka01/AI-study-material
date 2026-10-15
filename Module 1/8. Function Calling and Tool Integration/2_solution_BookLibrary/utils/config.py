import os

def load_system_prompt():
    """Load the system prompt from prompt.txt file."""
    try:
        # Look for prompt.txt in the solution directory (same level as main.py)
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompt.txt')
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError("prompt.txt file not found. Please ensure it exists in the solution directory")
    except Exception as e:
        raise Exception(f"Error loading prompt.txt: {str(e)}")

def get_system_prompt():
    """Get the system prompt for the restaurant assistant."""
    return load_system_prompt()

# Advanced restaurant database for demonstration
RESTAURANT_DATA = {
    "menu_items": {
        "SAL001": {
            "name": "Caesar Salad",
            "category": "appetizer",
            "price": 12.99,
            "dietary_restrictions": ["vegetarian"],
            "description": "Fresh romaine lettuce with Caesar dressing, parmesan, and croutons",
            "popularity": 95,
            "available": True
        },
        "SAL002": {
            "name": "Garden Salad",
            "category": "appetizer",
            "price": 10.99,
            "dietary_restrictions": ["vegetarian", "vegan", "gluten_free"],
            "description": "Mixed greens with seasonal vegetables",
            "popularity": 78,
            "available": True
        },
        "MAIN001": {
            "name": "Grilled Salmon",
            "category": "main_course", 
            "price": 28.99,
            "dietary_restrictions": ["gluten_free"],
            "description": "Atlantic salmon with herbs, lemon, and seasonal vegetables",
            "popularity": 88,
            "available": True
        },
        "MAIN002": {
            "name": "Vegetarian Pasta",
            "category": "main_course",
            "price": 22.99,
            "dietary_restrictions": ["vegetarian"],
            "description": "Fresh pasta with seasonal vegetables and marinara sauce",
            "popularity": 82,
            "available": True
        },
        "MAIN003": {
            "name": "Quinoa Bowl",
            "category": "main_course",
            "price": 18.99,
            "dietary_restrictions": ["vegetarian", "vegan", "gluten_free"],
            "description": "Quinoa with roasted vegetables and tahini dressing",
            "popularity": 75,
            "available": True
        },
        "DESS001": {
            "name": "Chocolate Cake",
            "category": "dessert",
            "price": 8.99,
            "dietary_restrictions": ["vegetarian"],
            "description": "Rich chocolate cake with berry compote",
            "popularity": 92,
            "available": True
        },
        "BEV001": {
            "name": "Fresh Juice",
            "category": "beverage",
            "price": 4.99,
            "dietary_restrictions": ["vegetarian", "vegan", "gluten_free"],
            "description": "Daily selection of fresh pressed juices",
            "popularity": 65,
            "available": True
        }
    },
    "tables": {
        "availability": {
            "2024-10-16": {
                "17:00": {"available": True, "capacity": 8},
                "17:30": {"available": True, "capacity": 6},
                "18:00": {"available": True, "capacity": 4},
                "18:30": {"available": False, "capacity": 0},
                "19:00": {"available": True, "capacity": 6},
                "19:30": {"available": False, "capacity": 0},
                "20:00": {"available": True, "capacity": 8},
                "20:30": {"available": True, "capacity": 4}
            },
            "2024-10-17": {
                "17:00": {"available": True, "capacity": 10},
                "18:00": {"available": True, "capacity": 8},
                "19:00": {"available": True, "capacity": 6},
                "20:00": {"available": True, "capacity": 4}
            }
        }
    },
    "customers": {
        "555-0789": {
            "name": "Jane Smith",
            "email": "jane@email.com",
            "birthday": "1985-06-15",
            "dietary_preferences": ["vegetarian"],
            "favorite_dishes": ["Caesar Salad", "Vegetarian Pasta"],
            "allergies": ["nuts"],
            "loyalty_points": 150
        },
        "555-0456": {
            "name": "Sarah Johnson",
            "email": "sarah@email.com",
            "birthday": "1990-03-22",
            "dietary_preferences": ["gluten_free"],
            "favorite_dishes": ["Grilled Salmon"],
            "allergies": [],
            "loyalty_points": 75
        }
    },
    "reservations": {
        "RES001": {
            "customer_name": "John Doe",
            "customer_phone": "555-0123",
            "date": "2024-10-16",
            "time": "19:00",
            "party_size": 4,
            "special_requests": "Window table preferred"
        }
    },
    "orders": {
        "ORD001": {
            "customer_info": {"name": "Sarah", "phone": "555-0456", "table_number": 5},
            "items": [
                {"menu_item_id": "SAL001", "quantity": 2, "special_instructions": ""},
                {"menu_item_id": "MAIN001", "quantity": 1, "special_instructions": "no herbs"}
            ],
            "total": 54.97,
            "status": "preparing",
            "discount_code": None
        }
    },
    "daily_specials": [
        {
            "name": "Chef's Special Risotto",
            "price": 26.99,
            "description": "Wild mushroom risotto with truffle oil",
            "dietary_restrictions": ["vegetarian"]
        },
        {
            "name": "Catch of the Day",
            "price": 32.99,
            "description": "Fresh fish with seasonal preparation",
            "dietary_restrictions": ["gluten_free"]
        }
    ]
}

# Advanced mock function implementations
def manage_reservation_mock(action, **kwargs):
    """Advanced mock function for reservation management"""
    if action == "create":
        reservation_id = f"RES{len(RESTAURANT_DATA['reservations']) + 1:03d}"
        
        # Check availability
        date = kwargs.get('date')
        time = kwargs.get('time')
        party_size = kwargs.get('party_size', 2)
        
        if date in RESTAURANT_DATA['tables']['availability']:
            if time in RESTAURANT_DATA['tables']['availability'][date]:
                slot = RESTAURANT_DATA['tables']['availability'][date][time]
                if not slot['available'] or slot['capacity'] < party_size:
                    return {
                        "success": False,
                        "message": f"No table available for {party_size} people on {date} at {time}"
                    }
        
        # Create reservation
        reservation_data = {
            "customer_name": kwargs.get('customer_name'),
            "customer_phone": kwargs.get('customer_phone'),
            "date": date,
            "time": time,
            "party_size": party_size,
            "special_requests": kwargs.get('special_requests', '')
        }
        
        RESTAURANT_DATA['reservations'][reservation_id] = reservation_data
        
        return {
            "success": True,
            "reservation_id": reservation_id,
            "message": f"Reservation created for {kwargs.get('customer_name')} on {date} at {time} for {party_size} people"
        }
    
    elif action == "modify":
        reservation_id = kwargs.get('reservation_id')
        if reservation_id not in RESTAURANT_DATA['reservations']:
            return {"success": False, "message": "Reservation not found"}
        
        # Update reservation
        reservation = RESTAURANT_DATA['reservations'][reservation_id]
        for key, value in kwargs.items():
            if key != 'action' and key != 'reservation_id':
                reservation[key] = value
        
        return {"success": True, "message": f"Reservation {reservation_id} updated successfully"}
    
    elif action == "cancel":
        reservation_id = kwargs.get('reservation_id')
        if reservation_id not in RESTAURANT_DATA['reservations']:
            return {"success": False, "message": "Reservation not found"}
        
        del RESTAURANT_DATA['reservations'][reservation_id]
        return {"success": True, "message": f"Reservation {reservation_id} cancelled successfully"}
    
    elif action == "check":
        date = kwargs.get('date')
        time = kwargs.get('time')
        party_size = kwargs.get('party_size', 2)
        
        if date in RESTAURANT_DATA['tables']['availability']:
            if time in RESTAURANT_DATA['tables']['availability'][date]:
                slot = RESTAURANT_DATA['tables']['availability'][date][time]
                available = slot['available'] and slot['capacity'] >= party_size
                return {
                    "available": available,
                    "message": f"Table for {party_size} {'available' if available else 'not available'} on {date} at {time}",
                    "capacity": slot['capacity'] if available else 0
                }
        
        return {"available": False, "message": "No availability found for requested time"}
    
    return {"success": False, "message": "Invalid reservation action"}

def search_menu_mock(category, **kwargs):
    """Advanced mock function for menu search with complex filtering"""
    results = []
    
    for item_id, item in RESTAURANT_DATA['menu_items'].items():
        # Filter by category
        if category != "all" and item['category'] != category:
            continue
        
        # Filter by availability
        if not item.get('available', True):
            continue
            
        # Filter by dietary restrictions
        dietary_restrictions = kwargs.get('dietary_restrictions', [])
        if dietary_restrictions:
            if not any(dr in item['dietary_restrictions'] for dr in dietary_restrictions):
                continue
        
        # Filter by price range
        price_range = kwargs.get('price_range', {})
        if price_range:
            min_price = price_range.get('min', 0)
            max_price = price_range.get('max', float('inf'))
            if not (min_price <= item['price'] <= max_price):
                continue
        
        # Filter by search term
        search_term = kwargs.get('search_term', '').lower()
        if search_term:
            if search_term not in item['name'].lower() and search_term not in item['description'].lower():
                continue
            
        results.append({
            "item_id": item_id,
            "name": item['name'],
            "price": item['price'],
            "category": item['category'],
            "dietary_restrictions": item['dietary_restrictions'],
            "description": item['description'],
            "popularity": item['popularity']
        })
    
    # Sort results
    sort_by = kwargs.get('sort_by', 'name')
    if sort_by == 'price_low':
        results.sort(key=lambda x: x['price'])
    elif sort_by == 'price_high':
        results.sort(key=lambda x: x['price'], reverse=True)
    elif sort_by == 'popularity':
        results.sort(key=lambda x: x['popularity'], reverse=True)
    else:  # name
        results.sort(key=lambda x: x['name'])
    
    return {"items": results, "count": len(results)}

def manage_order_mock(action, **kwargs):
    """Advanced mock function for order management"""
    if action == "create":
        order_id = f"ORD{len(RESTAURANT_DATA['orders']) + 1:03d}"
        
        order_data = {
            "customer_info": kwargs.get('customer_info', {}),
            "items": [],
            "total": 0,
            "status": "open",
            "discount_code": None
        }
        
        RESTAURANT_DATA['orders'][order_id] = order_data
        
        customer_name = kwargs.get('customer_info', {}).get('name', 'customer')
        return {
            "success": True,
            "order_id": order_id,
            "message": f"New order created for {customer_name}"
        }
    
    elif action == "add_item":
        order_id = kwargs.get('order_id')
        items = kwargs.get('items', [])
        
        if order_id not in RESTAURANT_DATA['orders']:
            return {"success": False, "message": "Order not found"}
        
        order = RESTAURANT_DATA['orders'][order_id]
        for item in items:
            order['items'].append(item)
        
        return {"success": True, "message": f"Items added to order {order_id}"}
    
    elif action == "calculate_total":
        order_id = kwargs.get('order_id')
        discount_code = kwargs.get('discount_code')
        
        if order_id not in RESTAURANT_DATA['orders']:
            return {"success": False, "message": "Order not found"}
        
        order = RESTAURANT_DATA['orders'][order_id]
        
        # Calculate subtotal
        subtotal = 0
        for item in order['items']:
            menu_item_id = item['menu_item_id']
            quantity = item['quantity']
            if menu_item_id in RESTAURANT_DATA['menu_items']:
                price = RESTAURANT_DATA['menu_items'][menu_item_id]['price']
                subtotal += price * quantity
        
        # Apply discount
        discount_rate = 0
        if discount_code == "BIRTHDAY10":
            discount_rate = 0.10
        elif discount_code == "WELCOME20":
            discount_rate = 0.20
        
        discount_amount = subtotal * discount_rate
        
        # Calculate tax
        tax_rate = 0.08
        tax_amount = (subtotal - discount_amount) * tax_rate
        
        total = subtotal - discount_amount + tax_amount
        
        # Update order
        order['total'] = total
        order['discount_code'] = discount_code
        
        return {
            "success": True,
            "subtotal": subtotal,
            "discount_amount": discount_amount,
            "tax_amount": tax_amount,
            "total": total,
            "message": f"Order total calculated: ${total:.2f}"
        }
    
    return {"success": False, "message": "Order action not implemented"}

def manage_customer_mock(action, phone, **kwargs):
    """Advanced mock function for customer management"""
    if action == "lookup":
        if phone in RESTAURANT_DATA['customers']:
            return {
                "found": True,
                "customer": RESTAURANT_DATA['customers'][phone]
            }
        else:
            return {"found": False, "message": "Customer not found"}
    
    elif action == "create":
        customer_data = kwargs.get('customer_data', {})
        RESTAURANT_DATA['customers'][phone] = customer_data
        return {"success": True, "message": f"Customer profile created for {customer_data.get('name')}"}
    
    elif action == "update_preferences":
        if phone not in RESTAURANT_DATA['customers']:
            return {"success": False, "message": "Customer not found"}
        
        customer_data = kwargs.get('customer_data', {})
        customer = RESTAURANT_DATA['customers'][phone]
        
        for key, value in customer_data.items():
            customer[key] = value
        
        return {"success": True, "message": "Customer preferences updated"}
    
    elif action == "add_loyalty_points":
        if phone not in RESTAURANT_DATA['customers']:
            return {"success": False, "message": "Customer not found"}
        
        points = kwargs.get('loyalty_points', 0)
        RESTAURANT_DATA['customers'][phone]['loyalty_points'] += points
        
        return {"success": True, "message": f"Added {points} loyalty points"}
    
    elif action == "check_points":
        if phone not in RESTAURANT_DATA['customers']:
            return {"success": False, "message": "Customer not found"}
        
        points = RESTAURANT_DATA['customers'][phone]['loyalty_points']
        return {"success": True, "points": points, "message": f"Customer has {points} loyalty points"}
    
    return {"success": False, "message": "Customer action not implemented"}

def get_restaurant_status_mock(info_type, **kwargs):
    """Advanced mock function for restaurant status"""
    if info_type == "wait_time":
        # Simulate different wait times based on time of day
        import datetime
        current_hour = datetime.datetime.now().hour
        
        if 17 <= current_hour <= 19:  # Peak dinner hours
            wait_time = 25
        elif 12 <= current_hour <= 14:  # Lunch rush
            wait_time = 15
        else:
            wait_time = 5
        
        return {"wait_time_minutes": wait_time, "message": f"Current wait time is approximately {wait_time} minutes"}
    
    elif info_type == "daily_specials":
        return {
            "specials": RESTAURANT_DATA["daily_specials"],
            "message": f"We have {len(RESTAURANT_DATA['daily_specials'])} daily specials available"
        }
    
    elif info_type == "table_availability":
        date = kwargs.get('date')
        time_slot = kwargs.get('time_slot')
        party_size = kwargs.get('party_size', 2)
        
        if date and time_slot:
            return manage_reservation_mock("check", date=date, time=time_slot, party_size=party_size)
        else:
            # Return general availability
            available_slots = []
            for date, times in RESTAURANT_DATA['tables']['availability'].items():
                for time, slot in times.items():
                    if slot['available'] and slot['capacity'] >= party_size:
                        available_slots.append(f"{date} at {time}")
            
            if available_slots:
                return {"message": f"Available time slots for {party_size} people: {', '.join(available_slots[:3])}..."}
            else:
                return {"message": "No available time slots found"}
    
    elif info_type == "kitchen_status":
        return {"message": "Kitchen is operating normally with standard wait times"}
    
    elif info_type == "operating_hours":
        return {"message": "We are open Monday-Sunday: 11:00 AM - 10:00 PM"}
    
    return {"message": f"Status information for {info_type} not available"}

# Export the system prompt
SYSTEM_PROMPT = get_system_prompt()
