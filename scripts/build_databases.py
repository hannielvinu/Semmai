"""Build the five SQLite databases used by the Semmai BIRD-style benchmark.

Each database is generated deterministically (fixed random seed) so every run
produces identical data and identical evaluation numbers.

Output:
    data/bird_style/databases/<db_id>/<db_id>.sqlite
    data/bird_style/schema_meta.json   (column descriptions + synonyms, like
                                        BIRD's database_description files)

Usage:
    python scripts/build_databases.py
"""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

if hasattr(__import__("sys").stdout, "reconfigure"):
    __import__("sys").stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bird_style"

FIRST = ["Arun", "Priya", "Karthik", "Divya", "Suresh", "Lakshmi", "Vignesh", "Anitha",
         "Rahul", "Meena", "Ganesh", "Kavya", "Sanjay", "Deepa", "Harish", "Revathi",
         "Ajay", "Sneha", "Manoj", "Pooja", "Vikram", "Nandhini", "Ravi", "Shalini",
         "Ashok", "Keerthana", "Naveen", "Janani", "Prakash", "Swathi", "Rohit", "Aishwarya",
         "Imran", "Fathima", "Joseph", "Mary", "David", "Sara", "Mohan", "Gayathri"]
LAST = ["Kumar", "Raman", "Iyer", "Nair", "Reddy", "Sharma", "Pillai", "Menon", "Rao",
        "Krishnan", "Subramanian", "Das", "Singh", "Patel", "Joseph", "Thomas", "Khan",
        "Murugan", "Selvam", "Balaji"]
CITIES = ["Chennai", "Vellore", "Coimbatore", "Madurai", "Bengaluru", "Hyderabad",
          "Mumbai", "Delhi", "Kolkata", "Pune", "Salem", "Tiruchirappalli"]


def rdate(rng: random.Random, start: date, end: date) -> str:
    return (start + timedelta(days=rng.randint(0, (end - start).days))).isoformat()


def person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def money(rng: random.Random, lo: float, hi: float, step: int = 50) -> float:
    return float(round(rng.uniform(lo, hi) / step) * step)


# ---------------------------------------------------------------- hospital
def build_hospital(rng):
    ddl = """
    CREATE TABLE patients (patient_id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
        gender TEXT, city TEXT, blood_group TEXT, ward TEXT, admission_date DATE,
        bill_amount REAL);
    CREATE TABLE doctors (doctor_id INTEGER PRIMARY KEY, name TEXT, specialty TEXT,
        experience_years INTEGER, city TEXT, consultation_fee REAL);
    CREATE TABLE appointments (appointment_id INTEGER PRIMARY KEY, patient_id INTEGER,
        doctor_id INTEGER, appointment_date DATE, status TEXT, fee REAL,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id),
        FOREIGN KEY(doctor_id) REFERENCES doctors(doctor_id));
    """
    rows = {"patients": [], "doctors": [], "appointments": []}
    wards = ["General", "ICU", "Pediatric", "Maternity", "Orthopedic", "Cardiac"]
    for i in range(1, 401):
        age = rng.randint(1, 95)
        ward = "Pediatric" if age < 14 else rng.choice([w for w in wards if w != "Pediatric"])
        rows["patients"].append((i, person(rng), age, rng.choice(["Male", "Female"]),
                                 rng.choice(CITIES), rng.choice(["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]),
                                 ward, rdate(rng, date(2019, 1, 1), date(2025, 12, 31)),
                                 money(rng, 1500, 250000, 100)))
    specs = ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "Dermatology",
             "General Medicine", "Oncology", "ENT"]
    for i in range(1, 81):
        rows["doctors"].append((i, "Dr. " + person(rng), rng.choice(specs), rng.randint(1, 38),
                                rng.choice(CITIES), money(rng, 300, 2500, 50)))
    for i in range(1, 901):
        rows["appointments"].append((i, rng.randint(1, 400), rng.randint(1, 80),
                                     rdate(rng, date(2021, 1, 1), date(2025, 12, 31)),
                                     rng.choice(["Completed", "Cancelled", "Scheduled", "No-show"]),
                                     money(rng, 300, 3000, 50)))
    return ddl, rows


# ---------------------------------------------------------------- retail
def build_retail(rng):
    ddl = """
    CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER,
        city TEXT, loyalty_tier TEXT, join_date DATE);
    CREATE TABLE products (product_id INTEGER PRIMARY KEY, name TEXT, category TEXT,
        price REAL, stock INTEGER, rating REAL);
    CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER,
        order_date DATE, amount REAL, payment_method TEXT, status TEXT,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id));
    """
    rows = {"customers": [], "products": [], "orders": []}
    for i in range(1, 351):
        rows["customers"].append((i, person(rng), rng.randint(18, 75), rng.choice(CITIES),
                                  rng.choice(["Bronze", "Silver", "Gold", "Platinum"]),
                                  rdate(rng, date(2018, 1, 1), date(2025, 12, 31))))
    cats = {"Electronics": ["Wireless Earbuds", "Smart Watch", "Bluetooth Speaker", "Power Bank",
                            "LED Monitor", "Mechanical Keyboard", "Gaming Mouse", "Tablet"],
            "Clothing": ["Cotton Kurta", "Denim Jacket", "Silk Saree", "Running Shoes",
                         "Formal Shirt", "Woollen Sweater"],
            "Groceries": ["Basmati Rice", "Filter Coffee Powder", "Organic Honey", "Cold Pressed Oil",
                          "Masala Tea", "Millet Mix"],
            "Books": ["Python Handbook", "Tamil Classics", "Data Science Primer", "History of India",
                      "Children's Atlas"],
            "Furniture": ["Study Table", "Office Chair", "Bookshelf", "Teak Wood Bed", "Sofa Set"],
            "Toys": ["Building Blocks", "Remote Car", "Puzzle Set", "Board Game"]}
    pid = 0
    for cat, names in cats.items():
        for n in names:
            for variant in ["", " Pro", " Lite"]:
                pid += 1
                base = {"Electronics": (800, 45000), "Clothing": (400, 9000), "Groceries": (60, 1500),
                        "Books": (150, 1800), "Furniture": (2500, 60000), "Toys": (200, 4500)}[cat]
                rows["products"].append((pid, n + variant, cat, money(rng, *base, 10),
                                         rng.randint(0, 500), round(rng.uniform(2.5, 5.0), 1)))
    for i in range(1, 1201):
        rows["orders"].append((i, rng.randint(1, 350), rdate(rng, date(2020, 1, 1), date(2025, 12, 31)),
                               money(rng, 150, 60000, 10),
                               rng.choice(["UPI", "Card", "Cash", "NetBanking"]),
                               rng.choice(["Delivered", "Shipped", "Cancelled", "Returned"])))
    return ddl, rows


# ---------------------------------------------------------------- university
def build_university(rng):
    ddl = """
    CREATE TABLE students (student_id INTEGER PRIMARY KEY, name TEXT, department TEXT,
        year_of_study INTEGER, cgpa REAL, city TEXT, enrollment_date DATE, age INTEGER);
    CREATE TABLE courses (course_id INTEGER PRIMARY KEY, title TEXT, department TEXT,
        credits INTEGER);
    CREATE TABLE enrollments (enrollment_id INTEGER PRIMARY KEY, student_id INTEGER,
        course_id INTEGER, semester TEXT, marks INTEGER,
        FOREIGN KEY(student_id) REFERENCES students(student_id),
        FOREIGN KEY(course_id) REFERENCES courses(course_id));
    """
    rows = {"students": [], "courses": [], "enrollments": []}
    depts = ["CSE", "ECE", "EEE", "Mechanical", "Civil", "IT", "Biotech"]
    for i in range(1, 501):
        yr = rng.randint(1, 4)
        rows["students"].append((i, person(rng), rng.choice(depts), yr,
                                 round(rng.uniform(5.0, 10.0), 2), rng.choice(CITIES),
                                 rdate(rng, date(2020 + 4 - yr, 6, 1), date(2020 + 4 - yr, 8, 31)),
                                 17 + yr + rng.randint(0, 2)))
    titles = {"CSE": ["Data Structures", "Operating Systems", "Machine Learning", "Compiler Design",
                      "Database Systems", "Natural Language Processing"],
              "ECE": ["Signals and Systems", "VLSI Design", "Digital Communication", "Embedded Systems"],
              "EEE": ["Power Systems", "Electrical Machines", "Control Systems"],
              "Mechanical": ["Thermodynamics", "Fluid Mechanics", "Machine Design"],
              "Civil": ["Structural Analysis", "Surveying", "Geotechnical Engineering"],
              "IT": ["Web Technologies", "Cloud Computing", "Information Security"],
              "Biotech": ["Genetics", "Bioinformatics", "Microbiology"]}
    cid = 0
    for d, ts in titles.items():
        for t in ts:
            cid += 1
            rows["courses"].append((cid, t, d, rng.choice([2, 3, 4, 5])))
    for i in range(1, 1501):
        rows["enrollments"].append((i, rng.randint(1, 500), rng.randint(1, cid),
                                    rng.choice(["2023-Odd", "2023-Even", "2024-Odd", "2024-Even", "2025-Odd"]),
                                    rng.randint(20, 100)))
    return ddl, rows


# ---------------------------------------------------------------- finance
def build_finance(rng):
    ddl = """
    CREATE TABLE accounts (account_id INTEGER PRIMARY KEY, holder_name TEXT, branch TEXT,
        account_type TEXT, balance REAL, opened_date DATE, holder_age INTEGER);
    CREATE TABLE transactions (transaction_id INTEGER PRIMARY KEY, account_id INTEGER,
        transaction_date DATE, amount REAL, transaction_type TEXT, channel TEXT,
        FOREIGN KEY(account_id) REFERENCES accounts(account_id));
    """
    rows = {"accounts": [], "transactions": []}
    for i in range(1, 451):
        rows["accounts"].append((i, person(rng), rng.choice(CITIES),
                                 rng.choice(["Savings", "Current", "Fixed Deposit", "Salary"]),
                                 money(rng, 0, 1500000, 100), rdate(rng, date(2010, 1, 1), date(2025, 12, 31)),
                                 rng.randint(18, 85)))
    for i in range(1, 2001):
        rows["transactions"].append((i, rng.randint(1, 450), rdate(rng, date(2022, 1, 1), date(2025, 12, 31)),
                                     money(rng, 50, 200000, 10), rng.choice(["Credit", "Debit"]),
                                     rng.choice(["UPI", "ATM", "Branch", "NetBanking", "Cheque"])))
    return ddl, rows


# ---------------------------------------------------------------- travel
def build_travel(rng):
    ddl = """
    CREATE TABLE flights (flight_id INTEGER PRIMARY KEY, flight_no TEXT, airline TEXT,
        origin TEXT, destination TEXT, departure_date DATE, duration_minutes INTEGER,
        price REAL, seats_available INTEGER);
    CREATE TABLE bookings (booking_id INTEGER PRIMARY KEY, flight_id INTEGER,
        passenger_name TEXT, passenger_age INTEGER, seat_class TEXT, booking_date DATE,
        amount REAL, FOREIGN KEY(flight_id) REFERENCES flights(flight_id));
    """
    rows = {"flights": [], "bookings": []}
    airlines = {"IndiGo": "6E", "Air India": "AI", "Vistara": "UK", "SpiceJet": "SG", "Akasa Air": "QP"}
    ports = ["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Kolkata", "Pune", "Coimbatore",
             "Madurai", "Kochi"]
    for i in range(1, 301):
        al = rng.choice(list(airlines))
        o, d = rng.sample(ports, 2)
        rows["flights"].append((i, f"{airlines[al]}-{rng.randint(100, 999)}", al, o, d,
                                rdate(rng, date(2024, 1, 1), date(2026, 6, 30)),
                                rng.randint(55, 190), money(rng, 2200, 18000, 10), rng.randint(0, 180)))
    for i in range(1, 1101):
        cls = rng.choice(["Economy", "Economy", "Economy", "Business", "Premium Economy"])
        rows["bookings"].append((i, rng.randint(1, 300), person(rng), rng.randint(1, 88), cls,
                                 rdate(rng, date(2023, 10, 1), date(2026, 6, 1)),
                                 money(rng, 2200, 45000 if cls == "Business" else 18000, 10)))
    return ddl, rows


# ---------------------------------------------------------------- hr (the project story)
def build_hr(rng):
    ddl = """
    CREATE TABLE departments (dept_name TEXT PRIMARY KEY, head TEXT, location TEXT, budget REAL);
    CREATE TABLE employees (emp_id INTEGER PRIMARY KEY, name TEXT, gender TEXT, age INTEGER,
        department TEXT, role TEXT, city TEXT, salary REAL, join_date DATE, experience_years INTEGER,
        FOREIGN KEY(department) REFERENCES departments(dept_name));
    """
    rows = {"departments": [], "employees": []}
    depts = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Support"]
    for d in depts:
        rows["departments"].append((d, person(rng), rng.choice(CITIES[:6]), money(rng, 2_000_000, 20_000_000, 100000)))
    roles = [("Intern", (12000, 25000), (0, 1)), ("Engineer", (40000, 120000), (1, 12)),
             ("Analyst", (35000, 90000), (1, 10)), ("Executive", (25000, 60000), (0, 8)),
             ("Team Lead", (90000, 180000), (5, 15)), ("Manager", (120000, 300000), (8, 25))]
    weights = [12, 30, 18, 20, 12, 8]
    for i in range(1, 601):
        role, (lo, hi), (elo, ehi) = rng.choices(roles, weights=weights)[0]
        exp = rng.randint(elo, ehi)
        age = max(21, min(60, 21 + exp + rng.randint(0, 6)))
        dept = rng.choice(depts)
        if role == "Engineer" and rng.random() < 0.7:
            dept = "Engineering"
        rows["employees"].append((i, person(rng), rng.choice(["Male", "Female"]), age, dept, role,
                                  rng.choice(CITIES), money(rng, lo, hi, 500),
                                  rdate(rng, date(2015, 1, 1), date(2025, 12, 31)), exp))
    return ddl, rows


# --------------------------------------------------------- schema metadata
# Human-written descriptions and synonyms (BIRD ships similar
# database_description CSV files). Semmai uses them for schema linking.
META = {
    "hr": {
        "employees": {"noun": ["employee", "employees", "staff", "worker", "workers", "people", "person"],
                      "name_col": "name",
                      "cols": {"name": ["name", "names"], "gender": ["gender", "sex"],
                               "age": ["age", "aged", "years old", "old"],
                               "department": ["department", "dept", "team"],
                               "role": ["role", "designation", "position", "post", "job title"],
                               "city": ["city", "from", "based", "located", "living", "lives"],
                               "salary": ["salary", "salaries", "pay", "earning", "earns", "paid", "income"],
                               "join_date": ["joined", "join date", "joining date", "joining", "hired"],
                               "experience_years": ["experience", "years of experience", "experienced"]}},
        "departments": {"noun": ["department", "departments"], "name_col": "dept_name",
                        "cols": {"dept_name": ["department name", "name"], "head": ["head"],
                                 "location": ["location", "located"], "budget": ["budget"]}},
    },
    "hospital": {
        "patients": {"noun": ["patient", "patients"], "name_col": "name",
                     "cols": {"name": ["name", "names"], "age": ["age", "aged", "years old", "old"],
                              "gender": ["gender", "sex"], "city": ["city", "lives", "living", "from"],
                              "blood_group": ["blood group", "blood type"], "ward": ["ward"],
                              "admission_date": ["admitted", "admission", "admission date"],
                              "bill_amount": ["bill", "bill amount", "billed", "hospital bill"]}},
        "doctors": {"noun": ["doctor", "doctors", "physician", "physicians"], "name_col": "name",
                    "cols": {"name": ["name", "names"], "specialty": ["specialty", "speciality", "specialization"],
                             "experience_years": ["experience", "years of experience", "experienced"],
                             "city": ["city", "practising", "practicing", "based"],
                             "consultation_fee": ["consultation fee", "fee", "charge", "charges"]}},
        "appointments": {"noun": ["appointment", "appointments", "visit", "visits"], "name_col": "appointment_id",
                         "cols": {"appointment_date": ["appointment date", "scheduled", "booked"],
                                  "status": ["status"], "fee": ["fee", "fees", "paid"]}},
    },
    "retail": {
        "customers": {"noun": ["customer", "customers", "shopper", "shoppers", "buyer", "buyers"], "name_col": "name",
                      "cols": {"name": ["name", "names"], "age": ["age", "aged", "years old", "old"],
                               "city": ["city", "from", "living", "located"],
                               "loyalty_tier": ["loyalty tier", "tier", "membership", "loyalty"],
                               "join_date": ["joined", "join date", "signed up", "registered"]}},
        "products": {"noun": ["product", "products", "item", "items"], "name_col": "name",
                     "cols": {"name": ["name", "names"], "category": ["category"],
                              "price": ["price", "priced", "cost", "costing", "costs", "expensive", "cheap"],
                              "stock": ["stock", "units in stock", "inventory", "units"],
                              "rating": ["rating", "rated", "stars"]}},
        "orders": {"noun": ["order", "orders", "purchase", "purchases"], "name_col": "order_id",
                   "cols": {"order_date": ["ordered", "order date", "placed"],
                            "amount": ["amount", "purchases", "purchase", "worth", "value", "spent", "order value"],
                            "payment_method": ["payment method", "paid via", "paid using", "payment"],
                            "status": ["status"]}},
    },
    "university": {
        "students": {"noun": ["student", "students"], "name_col": "name",
                     "cols": {"name": ["name", "names"], "department": ["department", "branch", "dept"],
                              "year_of_study": ["year of study", "study year"],
                              "cgpa": ["cgpa", "gpa", "grade point"], "city": ["city", "hometown", "from"],
                              "enrollment_date": ["enrolled", "enrollment date", "admitted"],
                              "age": ["age", "aged", "years old", "old"]}},
        "courses": {"noun": ["course", "courses", "subject", "subjects"], "name_col": "title",
                    "cols": {"title": ["title", "titles", "name", "names"], "department": ["department", "offered by"],
                             "credits": ["credits", "credit", "credit hours"]}},
        "enrollments": {"noun": ["enrollment", "enrollments", "registration", "registrations"],
                        "name_col": "enrollment_id",
                        "cols": {"semester": ["semester", "term"], "marks": ["marks", "score", "scored", "mark"]}},
    },
    "finance": {
        "accounts": {"noun": ["account", "accounts", "account holder", "account holders"], "name_col": "holder_name",
                     "cols": {"holder_name": ["holder name", "holder", "name", "names"],
                              "branch": ["branch", "city", "from"], "account_type": ["account type", "type"],
                              "balance": ["balance", "deposit", "funds"],
                              "opened_date": ["opened", "open date", "created"],
                              "holder_age": ["age", "aged", "years old", "old"]}},
        "transactions": {"noun": ["transaction", "transactions", "payment", "payments", "transfer", "transfers"],
                         "name_col": "transaction_id",
                         "cols": {"transaction_date": ["transaction date", "made", "done"],
                                  "amount": ["amount", "worth", "value"],
                                  "transaction_type": ["transaction type", "type"],
                                  "channel": ["channel", "via", "through", "using"]}},
    },
    "travel": {
        "flights": {"noun": ["flight", "flights"], "name_col": "flight_no",
                    "cols": {"flight_no": ["flight number", "flight numbers", "flight no"],
                             "airline": ["airline", "carrier", "operated by"], "origin": ["origin", "from", "departing"],
                             "destination": ["destination", "to", "arriving"],
                             "departure_date": ["departure date", "departing", "departs", "leaving", "flying"],
                             "duration_minutes": ["duration", "minutes", "long", "flight time"],
                             "price": ["price", "fare", "cost", "costing", "priced", "ticket price"],
                             "seats_available": ["seats available", "available seats", "seats", "free seats"]}},
        "bookings": {"noun": ["booking", "bookings", "passenger", "passengers", "reservation", "reservations"],
                     "name_col": "passenger_name",
                     "cols": {"passenger_name": ["passenger name", "name", "names"],
                              "passenger_age": ["age", "aged", "years old", "old"],
                              "seat_class": ["class", "seat class", "cabin"],
                              "booking_date": ["booked", "booking date", "reserved"],
                              "amount": ["amount", "paid", "fare paid", "worth"]}},
    },
}

BUILDERS = {"hospital": build_hospital, "retail": build_retail, "university": build_university,
            "finance": build_finance, "travel": build_travel, "hr": build_hr}


def main() -> None:
    for i, (db_id, fn) in enumerate(BUILDERS.items()):
        rng = random.Random(2026 + i)
        ddl, rows = fn(rng)
        path = OUT / "databases" / db_id / f"{db_id}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        con.executescript(ddl)
        for table, data in rows.items():
            ph = ",".join("?" * len(data[0]))
            con.executemany(f"INSERT INTO {table} VALUES ({ph})", data)
        con.commit()
        con.close()
        print(f"built {db_id}: " + ", ".join(f"{t}={len(r)}" for t, r in rows.items()))
    (OUT / "schema_meta.json").write_text(json.dumps(META, indent=2), encoding="utf-8")
    print("wrote schema_meta.json")


if __name__ == "__main__":
    main()
