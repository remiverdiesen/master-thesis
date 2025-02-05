class Building:
    def __init__(self, surface_area, outer_wall_length, building_value_per_m2):
        """
        Initialize the building with basic structure information.

        :param surface_area:            Total surface area of the building in square meters.
        :param outer_wall_length:       Length of the outer wall in meters.
        :param building_value_per_m2:   The direct damage value per square meter of the building (€/m²).
        """
        self.surface_area = surface_area
        self.outer_wall_length = outer_wall_length
        self.building_value_per_m2 = building_value_per_m2
        


class WaterDamageEstimator:
    def __init__(self, water_depth_outside, threshold_height=0.15):
        """
        Initialize the water damage estimator with water depth information.

        :param water_depth_outside:     Depth of water outside the building in meters.
        :param threshold_height:        Height of the threshold above the ground level where water starts to enter the building.
                                        Default value is 0.15 meters.
        """
        self.water_depth_outside = water_depth_outside
        self.threshold_height = threshold_height
        self.damage_amounts = self.get_damage_amounts()
        self.repair_estimates = self.get_repair_costs()
        
    def calculate_water_depth_inside(self):
        """
        Calculate the water depth inside the building based on the water depth outside and the threshold height.
        
        :return: Water depth inside the building in meters.
        """
        if self.water_depth_outside > self.threshold_height:
            return self.water_depth_outside - self.threshold_height
        else:
            return 0

    def calculate_damage_factor(self, water_depth_inside):
        """
        Calculate the damage factor based on the water depth inside the building. The damage factor is a value between
        0 and 1 that represents the percentage of maximum damage for a given water depth.

        :param water_depth_inside:  Water depth inside the building in meters.
        :return:                    Damage factor (a value between 0 and 1).
        """
        max_depth = 0.30  # Maximum depth at which full damage occurs
        if water_depth_inside >= max_depth:
            return 1  # Maximum damage factor
        elif water_depth_inside > 0:
            return water_depth_inside / max_depth  # Linear interpolation between 0 and max_depth
        else:
            return 0  # No damage if there's no water inside

    def calculate_damage(self, building):
        """
        Calculate the total direct damage to the building based on water depth and building characteristics.

        :param building:    An instance of the Building class.
        :return:            Total direct damage in euros (€).
        """
        water_depth_inside = self.calculate_water_depth_inside()
        damage_factor = self.calculate_damage_factor(water_depth_inside)
        total_damage = damage_factor * building.surface_area * building.building_value_per_m2
        return total_damage

    def get_damage_amounts(self):
        path = r'.\data\WSS\Schadebedragen - WSS.txt'
        with open(path, 'r') as file:
            damage_amounts = file.readlines()
        return damage_amounts
    
    def get_repair_costs(self):
        path = r'.\data\WSS\Herstelkosten - gechat.txt'
        with open(path, 'r') as file:
            repair_estimates = file.readlines()
        return repair_estimates
    

# Example Usage
if __name__ == "__main__":
    # Example input values
    water_depth_outside = 0.25   # Water depth outside the building in meters
    surface_area = 100           # Surface area of the building in square meters
    outer_wall_length = 40       # Length of the outer wall in meters: 
    building_value_per_m2 = 271  # Building value per m² in euros (€)

    # Create a Building object
    residential_building = Building(surface_area, outer_wall_length, building_value_per_m2 = 271)

    # Create a WaterDamageEstimator object
    damage_estimator = WaterDamageEstimator(water_depth_outside)

    # Calculate the direct damage
    total_direct_damage = damage_estimator.calculate_damage(residential_building)

    # Output the result
    print(f"Total direct damage to the building: €{total_direct_damage:.2f}")
