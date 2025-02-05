class WaterDamageEstimator:
    def __init__(self, water_depth, building_area, water_fraction, wall_length, threshold_height, building_type, has_basement, luck_factor):
        """
        Initialize the water damage estimator for a building.

        Parameters:
        - water_depth (float): Average water depth around the building (m).
        - building_area (float): Total floor surface area of the residential building (m²).
        - water_fraction (float): Fraction of the area around the building that is submerged in water.
        - wall_length (float): Total length of exterior walls of the building exposed to water (m).
        - threshold_height (float): Threshold height of the building’s doorway or other water entry points (m).
        - building_type (str): Type of building ('residential' for this case).
        - has_basement (bool): Whether the building has a basement.
        - luck_factor (float): Luck factor to account for mitigating actions (0-1 range).
        """
        self.water_depth = water_depth
        self.building_area = building_area
        self.water_fraction = water_fraction
        self.wall_length = wall_length
        self.threshold_height = threshold_height
        self.building_type = building_type
        self.has_basement = has_basement
        self.luck_factor = luck_factor

        # Set damage per square meter based on building type (as per TAUW method)
        if self.building_type == 'residential':
            self.damage_per_m2 = 250  # €/m² (STOWA 2013 average for residential buildings)

    def calculate_water_inflow_probability(self):
        """
        Calculate the probability that water will enter the building based on water fraction and wall exposure.
        """
        facade_fraction = self.water_fraction  # Assume facade fraction = water fraction for simplicity
        F_water_damage_normal = (self.water_fraction * (facade_fraction * 4)) / self.wall_length
        return F_water_damage_normal

    def adjust_for_basement(self, F_water_damage_normal):
        """
        Adjust water inflow probability if a basement is present.
        """
        if self.has_basement:
            return 1  # Full damage probability if basement exists
        else:
            return F_water_damage_normal

    def apply_luck_factor(self, F_damage_probability):
        """
        Apply the luck factor to adjust the damage probability.
        """
        return F_damage_probability * (1 - self.luck_factor)

    def calculate_water_inflow_area(self):
        """
        Calculate the amount of water inflow area based on water depth.
        """
        L_inflow = 20 * self.water_depth
        return min(L_inflow, self.building_area)  # Water cannot spread beyond the building area

    def calculate_total_direct_damage(self):
        """
        Calculate the total direct damage to the building in monetary terms (€).
        """
        # Step 1: Calculate the water inflow probability
        F_water_damage_normal = self.calculate_water_inflow_probability()

        # Step 2: Adjust for basement presence
        F_damage_probability = self.adjust_for_basement(F_water_damage_normal)

        # Step 3: Apply the luck factor
        F_damage_probability = self.apply_luck_factor(F_damage_probability)

        # Step 4: Calculate water inflow area
        L_inflow = self.calculate_water_inflow_area()

        # Step 5: Calculate the total direct damage
        total_damage = F_damage_probability * L_inflow * self.building_area * self.damage_per_m2
        return total_damage

# Example usage:
if __name__ == "__main__":
    # Sample data for the building:
    water_depth = 0.3  # meters
    building_area = 100  # m²
    water_fraction = 0.7  # 70% of the surrounding area submerged
    wall_length = 40  # meters
    threshold_height = 0.15  # meters
    building_type = 'residential'
    has_basement = False
    luck_factor = 0.1  # 10% luck factor

    # Create an instance of the WaterDamageEstimator class
    damage_estimator = WaterDamageEstimator(
        water_depth, building_area, water_fraction, wall_length, threshold_height, building_type, has_basement, luck_factor
    )

    # Calculate the total direct damage
    total_direct_damage = damage_estimator.calculate_total_direct_damage()

    print(f"Estimated Total Direct Damage: €{total_direct_damage:.2f}")
