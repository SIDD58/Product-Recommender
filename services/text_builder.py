from schemas.product_schema import Product

# Combine tags and title into a single text

def product_to_text(product: Product) -> str:
    tags = ", ".join(product.tags)
    return f"{product.title}. Tags: {tags}"