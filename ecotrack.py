import reflex as rx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 1. AI & MACHINE LEARNING MODEL ---
# This trains the AI to recognize bank descriptions
TRAINING_DATA = [
    ("Chevron Gas Station", "Transport"), ("Uber Trip", "Transport"), 
    ("Starbucks Coffee", "Food"), ("McDonalds", "Food"),
    ("Delta Airlines", "Travel"), ("Amazon Marketplace", "Shopping"),
    ("Shell Oil", "Transport"), ("Whole Foods", "Food")
]
texts, labels = zip(*TRAINING_DATA)
vectorizer = TfidfVectorizer()
classifier = MultinomialNB()
classifier.fit(vectorizer.fit_transform(texts), labels)

# Carbon Intensity (kg CO2 per $1)
EMISSION_FACTORS = {"Transport": 0.45, "Food": 0.25, "Travel": 0.80, "Shopping": 0.15}

# --- 2. BACKEND LOGIC (State) ---
class State(rx.State):
    description: str = ""
    amount: str = ""  # Input comes as string
    category: str = "Pending"
    co2_val: float = 0.0
    history: list[dict] = []

    def calculate(self):
        if not self.description or not self.amount:
            return
        
        # AI prediction
        test_vec = vectorizer.transform([self.description])
        self.category = str(classifier.predict(test_vec)[0])
        
        # Math calculation
        amt = float(self.amount)
        factor = EMISSION_FACTORS.get(self.category, 0.10)
        self.co2_val = round(amt * factor, 2)
        
        # Add to history list
        self.history = [{"desc": self.description, "cat": self.category, "co2": self.co2_val}] + self.history

# --- 3. FRONTEND UI (Website Design) ---
def index():
    return rx.center(
        rx.vstack(
            rx.heading("EcoTrack AI", size="8", color_scheme="leaf"),
            rx.text("Automated Carbon Tracking from Bank Data", color="gray"),
            
            rx.box(
                rx.vstack(
                    rx.input(placeholder="Bank Description (e.g. Uber)", on_blur=State.set_description, width="100%"),
                    rx.input(placeholder="Amount ($)", on_blur=State.set_amount, width="100%"),
                    rx.button("Analyze Impact", on_click=State.calculate, color_scheme="green", width="100%"),
                    spacing="3",
                ),
                padding="2em",
                border_radius="lg",
                border="1px solid #EAEAEA",
                width="100%",
            ),

            rx.cond(
                State.co2_val > 0,
                rx.card(
                    rx.stat_group(
                        rx.stat(
                            rx.stat_label("Estimated Category"),
                            rx.stat_number(State.category),
                        ),
                        rx.stat(
                            rx.stat_label("Carbon Footprint"),
                            rx.stat_number(f"{State.co2_val} kg"),
                            rx.stat_help_text("CO2e"),
                        ),
                    ),
                    width="100%",
                )
            ),

            rx.divider(),
            rx.heading("Recent Transactions", size="4"),
            rx.table.root(
                rx.table.header(
                    rx.table.row(
                        rx.table.column_header_cell("Description"),
                        rx.table.column_header_cell("Category"),
                        rx.table.column_header_cell("CO2 (kg)"),
                    )
                ),
                rx.table.body(
                    rx.foreach(State.history, lambda item: rx.table.row(
                        rx.table.cell(item["desc"]),
                        rx.table.cell(item["cat"]),
                        rx.table.cell(item["co2"]),
                    ))
                ),
                width="100%",
            ),
            spacing="5",
            width="450px",
            padding_y="5em",
        )
    )

app = rx.App()
app.add_page(index)