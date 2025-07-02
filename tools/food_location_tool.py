"""
Food Location Tool - Restaurant Finder
=======================================

Suggests good food places and restaurants near current location.
Perfect for discovering great local food spots and hidden gems.
"""

from typing import Optional

async def get_food_locations(latitude: Optional[float] = None, longitude: Optional[float] = None, food_type: Optional[str] = "all", budget_range: Optional[str] = "moderate") -> dict:
    """
    Suggests good food places and restaurants near current location.
    Perfect for discovering great local food spots and hidden gems.
    """
    # Sample restaurant data (in real implementation, would use location API)
    restaurant_database = [
        {
            "name": "Sharma Ji Ka Dhaba",
            "type": "North Indian",
            "speciality": "Dal Makhani, Butter Naan",
            "budget": "budget",
            "rating": 4.2,
            "distance": "0.5 km",
            "description": "Authentic homestyle North Indian food",
            "timings": "8:00 AM - 11:00 PM",
            "popular_dishes": ["Dal Makhani", "Paneer Butter Masala", "Garlic Naan"]
        },
        {
            "name": "South Spice Corner",
            "type": "South Indian", 
            "speciality": "Dosa, Idli Sambhar",
            "budget": "budget",
            "rating": 4.0,
            "distance": "0.8 km",
            "description": "Crispy dosas and authentic South Indian breakfast",
            "timings": "6:00 AM - 10:00 PM",
            "popular_dishes": ["Masala Dosa", "Rava Idli", "Filter Coffee"]
        },
        {
            "name": "Biryani Blues",
            "type": "Mughlai",
            "speciality": "Hyderabadi Biryani",
            "budget": "moderate",
            "rating": 4.5,
            "distance": "1.2 km", 
            "description": "Authentic Hyderabadi biryani with rich flavors",
            "timings": "12:00 PM - 11:00 PM",
            "popular_dishes": ["Chicken Biryani", "Mutton Biryani", "Raita"]
        },
        {
            "name": "Pizza Paradise",
            "type": "Italian",
            "speciality": "Wood-fired Pizza",
            "budget": "expensive",
            "rating": 4.3,
            "distance": "1.5 km",
            "description": "Authentic Italian pizzas with fresh ingredients",
            "timings": "11:00 AM - 11:30 PM",
            "popular_dishes": ["Margherita Pizza", "Pasta Alfredo", "Garlic Bread"]
        },
        {
            "name": "Chaat Gali",
            "type": "Street Food",
            "speciality": "Pani Puri, Bhel Puri",
            "budget": "budget",
            "rating": 4.1,
            "distance": "0.3 km",
            "description": "Best street food with authentic flavors",
            "timings": "4:00 PM - 10:00 PM",
            "popular_dishes": ["Pani Puri", "Sev Puri", "Aloo Tikki"]
        }
    ]
    
    # Filter by food type
    filtered_restaurants = restaurant_database
    if food_type and food_type != "all":
        filtered_restaurants = [r for r in restaurant_database if food_type.lower() in r["type"].lower()]
    
    # Filter by budget
    if budget_range:
        filtered_restaurants = [r for r in filtered_restaurants if r["budget"] == budget_range]
    
    # If no matches, return all restaurants
    if not filtered_restaurants:
        filtered_restaurants = restaurant_database
    
    # Sort by rating and distance
    filtered_restaurants.sort(key=lambda x: (-x["rating"], float(x["distance"].split()[0])))
    
    return {
        "search_criteria": {
            "location": f"({latitude}, {longitude})" if latitude and longitude else "Current Location",
            "food_type": food_type,
            "budget_range": budget_range
        },
        "recommended_places": filtered_restaurants[:5],  # Top 5 recommendations
        "discovery_tips": [
            "Check reviews and ratings before visiting",
            "Call ahead to confirm timings and availability", 
            "Try local specialties for authentic experience",
            "Consider ordering online for convenience"
        ],
        "note": "Distances are approximate. Actual travel time may vary based on traffic."
    } 