class utils:
    def safe_division(self, nominator, denominator):
        safe_division_constant = 0.0000001
        safe_divided = nominator/(denominator+safe_division_constant)
        return safe_divided