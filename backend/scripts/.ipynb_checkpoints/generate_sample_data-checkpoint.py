#!/usr/bin/env python3
"""
Generate synthetic sample data for the Context Graph demo.
Creates realistic financial institution data with decision traces.
"""

import json
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
from faker import Faker
from neo4j import GraphDatabase

load_dotenv()

fake = Faker()
Faker.seed(42)
random.seed(42)

# Configuration
NUM_PERSONS = 200
NUM_ACCOUNTS = 350
NUM_TRANSACTIONS = 2000
NUM_DECISIONS = 600
NUM_EMPLOYEES = 30
NUM_ORGANIZATIONS = 50
NUM_POLICIES = 15
NUM_SUPPORT_TICKETS = 100
NUM_ALERTS = 80

# Risk distribution
RISK_DISTRIBUTION = {"low": 0.60, "medium": 0.25, "high": 0.12, "critical": 0.03}

DECISION_TYPES = [
    ("approval", 0.40),
    ("rejection", 0.20),
    ("escalation", 0.18),
    ("exception", 0.12),
    ("override", 0.05),
    ("review", 0.05),
]

DECISION_CATEGORIES = [
    "credit",
    "fraud",
    "compliance",
    "trading",
    "support",
    "account_management",
]

ACCOUNT_TYPES = ["checking", "savings", "trading", "margin"]
TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer", "trade"]
CHANNELS = ["online", "branch", "atm", "wire", "mobile"]


class DataGenerator:
    def __init__(self, uri=None, username=None, password=None, database=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

        # Store generated IDs for relationships
        self.person_ids = []
        self.account_ids = []
        self.transaction_ids = []
        self.decision_ids = []
        self.employee_ids = []
        self.organization_ids = []
        self.policy_ids = []
        self.alert_ids = []
        self.support_ticket_ids = []

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Clear all data from the database."""
        print("Clearing existing data...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_constraints_and_indexes(self):
        """Create constraints and indexes."""
        print("Creating constraints and indexes...")
        with self.driver.session(database=self.database) as session:
            constraints = [
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT organization_id_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
                "CREATE CONSTRAINT employee_id_unique IF NOT EXISTS FOR (e:Employee) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT policy_id_unique IF NOT EXISTS FOR (p:Policy) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT support_ticket_id_unique IF NOT EXISTS FOR (s:SupportTicket) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:Alert) REQUIRE a.id IS UNIQUE",
            ]
            indexes = [
                "CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.normalized_name)",
                "CREATE INDEX person_email_idx IF NOT EXISTS FOR (p:Person) ON (p.email)",
                "CREATE INDEX account_number_idx IF NOT EXISTS FOR (a:Account) ON (a.account_number)",
                "CREATE INDEX decision_type_idx IF NOT EXISTS FOR (d:Decision) ON (d.decision_type, d.category)",
                "CREATE INDEX decision_timestamp_idx IF NOT EXISTS FOR (d:Decision) ON (d.decision_timestamp)",
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    pass  # Constraint may already exist
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    pass
        print("Constraints and indexes created.")

    def generate_organizations(self):
        """Generate organization nodes."""
        print(f"Generating {NUM_ORGANIZATIONS} organizations...")
        industries = ["Finance", "Technology", "Healthcare", "Energy", "Retail", "Manufacturing"]
        org_types = ["corporation", "bank", "broker", "vendor", "employer"]

        with self.driver.session(database=self.database) as session:
            for _ in range(NUM_ORGANIZATIONS):
                org_id = str(uuid.uuid4())
                self.organization_ids.append(org_id)
                name = fake.company()

                session.run(
                    """
                    CREATE (o:Organization {
                        id: $id,
                        name: $name,
                        normalized_name: toLower($name),
                        type: $type,
                        industry: $industry,
                        country: $country,
                        risk_rating: $risk_rating,
                        sanctions_status: $sanctions_status,
                        source_systems: $source_systems,
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": org_id,
                        "name": name,
                        "type": random.choice(org_types),
                        "industry": random.choice(industries),
                        "country": fake.country(),
                        "risk_rating": random.choice(["A", "B", "C", "D"]),
                        "sanctions_status": random.choices(
                            ["clear", "watchlist", "blocked"], weights=[0.9, 0.08, 0.02]
                        )[0],
                        "source_systems": random.sample(
                            ["CRM", "Trading", "Compliance"], k=random.randint(1, 3)
                        ),
                    },
                )
        print(f"Created {NUM_ORGANIZATIONS} organizations.")

    def connect_organizations(self):
        """Connect organizations to persons and accounts."""
        print("Connecting organizations to the graph...")
        with self.driver.session(database=self.database) as session:
            # Connect persons to organizations as employers (WORKS_FOR)
            for person_id in self.person_ids:
                if random.random() > 0.3:  # 70% of persons have an employer
                    org_id = random.choice(self.organization_ids)
                    session.run(
                        """
                        MATCH (p:Person {id: $person_id})
                        MATCH (o:Organization {id: $org_id})
                        CREATE (p)-[:WORKS_FOR {
                            start_date: date($start_date),
                            role: $role
                        }]->(o)
                        """,
                        {
                            "person_id": person_id,
                            "org_id": org_id,
                            "start_date": fake.date_between(
                                start_date="-10y", end_date="today"
                            ).isoformat(),
                            "role": random.choice(
                                ["Employee", "Contractor", "Executive", "Manager", "Director"]
                            ),
                        },
                    )

            # Connect some accounts to organizations (corporate accounts)
            corporate_accounts = random.sample(
                self.account_ids, k=int(len(self.account_ids) * 0.15)
            )
            for account_id in corporate_accounts:
                org_id = random.choice(self.organization_ids)
                session.run(
                    """
                    MATCH (a:Account {id: $account_id})
                    MATCH (o:Organization {id: $org_id})
                    CREATE (o)-[:OWNS_ACCOUNT]->(a)
                    """,
                    {"account_id": account_id, "org_id": org_id},
                )

            # Connect some transactions to organizations (as counterparties)
            org_transactions = random.sample(
                self.transaction_ids, k=int(len(self.transaction_ids) * 0.2)
            )
            for txn_id in org_transactions:
                org_id = random.choice(self.organization_ids)
                session.run(
                    """
                    MATCH (t:Transaction {id: $txn_id})
                    MATCH (o:Organization {id: $org_id})
                    CREATE (t)-[:COUNTERPARTY]->(o)
                    """,
                    {"txn_id": txn_id, "org_id": org_id},
                )

            # Connect decisions to organizations (decisions about orgs)
            org_decisions = random.sample(self.decision_ids, k=int(len(self.decision_ids) * 0.1))
            for decision_id in org_decisions:
                org_id = random.choice(self.organization_ids)
                session.run(
                    """
                    MATCH (d:Decision {id: $decision_id})
                    MATCH (o:Organization {id: $org_id})
                    CREATE (d)-[:ABOUT]->(o)
                    """,
                    {"decision_id": decision_id, "org_id": org_id},
                )

        print("Connected organizations to the graph.")

    def generate_employees(self):
        """Generate employee nodes."""
        print(f"Generating {NUM_EMPLOYEES} employees...")
        departments = ["Trading", "Compliance", "Risk", "Support", "Credit", "Operations"]
        roles = ["Analyst", "Senior Analyst", "Manager", "Director", "VP", "SVP"]

        with self.driver.session(database=self.database) as session:
            for i in range(NUM_EMPLOYEES):
                emp_id = str(uuid.uuid4())
                self.employee_ids.append(emp_id)

                session.run(
                    """
                    CREATE (e:Employee {
                        id: $id,
                        employee_id: $employee_id,
                        name: $name,
                        department: $department,
                        role: $role,
                        authorization_level: $auth_level,
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": emp_id,
                        "employee_id": f"EMP{str(i + 1).zfill(5)}",
                        "name": fake.name(),
                        "department": random.choice(departments),
                        "role": random.choice(roles),
                        "auth_level": random.randint(1, 5),
                    },
                )
        print(f"Created {NUM_EMPLOYEES} employees.")

    def generate_policies(self):
        """Generate policy nodes."""
        print(f"Generating {NUM_POLICIES} policies...")
        policies = [
            (
                "Credit Limit Policy",
                "credit",
                "Standard credit limits based on income and credit score",
            ),
            (
                "High-Value Transaction Review",
                "fraud",
                "Review required for transactions over $10,000",
            ),
            (
                "KYC Refresh Policy",
                "compliance",
                "Customer identity verification refresh every 2 years",
            ),
            ("Trading Limit Policy", "trading", "Maximum position sizes based on client tier"),
            (
                "AML Threshold Monitoring",
                "compliance",
                "Structured transaction detection thresholds",
            ),
            (
                "Customer Complaint Escalation",
                "support",
                "Escalation criteria for customer complaints",
            ),
            ("Account Freeze Protocol", "fraud", "Conditions for immediate account freeze"),
            ("Margin Call Requirements", "trading", "Margin maintenance and call thresholds"),
            ("Document Retention Policy", "compliance", "Record keeping requirements"),
            ("Wire Transfer Verification", "fraud", "Verification steps for outgoing wires"),
            ("New Account Risk Assessment", "credit", "Initial risk scoring for new accounts"),
            ("Sanctions Screening Policy", "compliance", "OFAC and global sanctions checks"),
            ("Dispute Resolution Timeline", "support", "Maximum resolution times by dispute type"),
            (
                "Position Concentration Limits",
                "trading",
                "Single-name and sector concentration limits",
            ),
            ("Exception Approval Authority", "credit", "Authority levels for policy exceptions"),
        ]

        with self.driver.session(database=self.database) as session:
            for name, category, description in policies:
                policy_id = str(uuid.uuid4())
                self.policy_ids.append(policy_id)

                session.run(
                    """
                    CREATE (p:Policy {
                        id: $id,
                        name: $name,
                        description: $description,
                        category: $category,
                        version: '1.0',
                        effective_date: date('2024-01-01'),
                        threshold_rules: $rules,
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": policy_id,
                        "name": name,
                        "description": description,
                        "category": category,
                        "rules": json.dumps({"threshold": random.randint(1000, 100000)}),
                    },
                )
        print(f"Created {NUM_POLICIES} policies.")

    def generate_persons(self):
        """Generate person nodes."""
        print(f"Generating {NUM_PERSONS} persons...")
        with self.driver.session(database=self.database) as session:
            for _ in range(NUM_PERSONS):
                person_id = str(uuid.uuid4())
                self.person_ids.append(person_id)
                name = fake.name()

                # Assign risk score with realistic distribution
                risk_score = max(0, min(1, random.gauss(0.25, 0.2)))

                session.run(
                    """
                    CREATE (p:Person {
                        id: $id,
                        name: $name,
                        normalized_name: toLower($name),
                        email: $email,
                        phone: $phone,
                        date_of_birth: date($dob),
                        risk_score: $risk_score,
                        source_systems: $source_systems,
                        created_at: datetime()
                    })
                    """,
                    {
                        "id": person_id,
                        "name": name,
                        "email": fake.email(),
                        "phone": fake.phone_number(),
                        "dob": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
                        "risk_score": round(risk_score, 3),
                        "source_systems": random.sample(
                            ["CRM", "Trading", "Support", "Core Banking"], k=random.randint(1, 3)
                        ),
                    },
                )
        print(f"Created {NUM_PERSONS} persons.")

    def generate_accounts(self):
        """Generate account nodes and link to persons."""
        print(f"Generating {NUM_ACCOUNTS} accounts...")
        with self.driver.session(database=self.database) as session:
            for i in range(NUM_ACCOUNTS):
                account_id = str(uuid.uuid4())
                self.account_ids.append(account_id)
                owner_id = random.choice(self.person_ids)

                # Risk tier based on account type and random factors
                risk_tier = random.choices(
                    list(RISK_DISTRIBUTION.keys()),
                    weights=list(RISK_DISTRIBUTION.values()),
                )[0]

                session.run(
                    """
                    CREATE (a:Account {
                        id: $id,
                        account_number: $account_number,
                        account_type: $account_type,
                        status: $status,
                        balance: $balance,
                        currency: 'USD',
                        risk_tier: $risk_tier,
                        opened_date: date($opened_date),
                        source_system: $source_system,
                        created_at: datetime()
                    })
                    WITH a
                    MATCH (p:Person {id: $owner_id})
                    CREATE (p)-[:OWNS]->(a)
                    """,
                    {
                        "id": account_id,
                        "account_number": f"ACC{str(i + 1).zfill(8)}",
                        "account_type": random.choice(ACCOUNT_TYPES),
                        "status": random.choices(
                            ["active", "frozen", "closed"], weights=[0.9, 0.05, 0.05]
                        )[0],
                        "balance": round(random.uniform(100, 500000), 2),
                        "risk_tier": risk_tier,
                        "opened_date": fake.date_between(
                            start_date="-5y", end_date="today"
                        ).isoformat(),
                        "source_system": random.choice(["Core Banking", "Trading"]),
                        "owner_id": owner_id,
                    },
                )
        print(f"Created {NUM_ACCOUNTS} accounts.")

    def generate_transactions(self):
        """Generate transaction nodes."""
        print(f"Generating {NUM_TRANSACTIONS} transactions...")
        with self.driver.session(database=self.database) as session:
            for i in range(NUM_TRANSACTIONS):
                txn_id = str(uuid.uuid4())
                self.transaction_ids.append(txn_id)
                from_account = random.choice(self.account_ids)
                to_account = random.choice(self.account_ids)

                # Generate amount with realistic distribution
                amount = round(random.lognormvariate(7, 2), 2)  # Log-normal distribution

                # Flag some transactions as suspicious
                status = random.choices(
                    ["completed", "pending", "flagged", "reversed"],
                    weights=[0.85, 0.08, 0.05, 0.02],
                )[0]

                risk_score = 0.1 if status == "completed" else 0.7 if status == "flagged" else 0.3

                session.run(
                    """
                    CREATE (t:Transaction {
                        id: $id,
                        transaction_id: $txn_id,
                        type: $type,
                        amount: $amount,
                        currency: 'USD',
                        timestamp: datetime($timestamp),
                        status: $status,
                        channel: $channel,
                        description: $description,
                        risk_score: $risk_score,
                        source_system: 'Core Banking',
                        created_at: datetime()
                    })
                    WITH t
                    MATCH (from:Account {id: $from_account})
                    MATCH (to:Account {id: $to_account})
                    CREATE (t)-[:FROM_ACCOUNT]->(from)
                    CREATE (t)-[:TO_ACCOUNT]->(to)
                    """,
                    {
                        "id": txn_id,
                        "txn_id": f"TXN{str(i + 1).zfill(10)}",
                        "type": random.choice(TRANSACTION_TYPES),
                        "amount": amount,
                        "timestamp": fake.date_time_between(
                            start_date="-1y", end_date="now"
                        ).isoformat(),
                        "status": status,
                        "channel": random.choice(CHANNELS),
                        "description": fake.sentence(nb_words=6),
                        "risk_score": round(risk_score, 3),
                        "from_account": from_account,
                        "to_account": to_account,
                    },
                )
        print(f"Created {NUM_TRANSACTIONS} transactions.")

    def generate_decisions(self):
        """Generate decision nodes with full reasoning traces."""
        print(f"Generating {NUM_DECISIONS} decisions...")

        reasoning_templates = {
            ("approval", "credit"): [
                "Credit application approved. Customer has credit score of {score}, stable income verified at ${income}/year, "
                "and debt-to-income ratio of {dti}%. Account has been in good standing for {months} months with no late payments. "
                "Approved credit limit increase to ${limit}.",
                "Manual review completed. While credit score of {score} is borderline, customer demonstrates {months} month "
                "relationship history with excellent payment behavior. Exception approved based on relationship value.",
            ],
            ("rejection", "credit"): [
                "Credit application rejected. Credit score of {score} below minimum threshold of 620. "
                "Recent derogatory marks including {marks} late payments in last 12 months. "
                "Debt-to-income ratio of {dti}% exceeds maximum allowed 45%.",
            ],
            ("approval", "fraud"): [
                "Transaction cleared after review. Initial fraud alert triggered due to {reason}. "
                "Customer verified via {method}. Transaction amount of ${amount} consistent with customer profile. "
                "No further action required.",
            ],
            ("rejection", "fraud"): [
                "Transaction blocked. Pattern matches known fraud typology FT-{code}. "
                "Velocity check failed: {count} transactions in {minutes} minutes. "
                "Geographic anomaly: IP location {location1} inconsistent with account address {location2}. "
                "Account flagged for investigation.",
            ],
            ("escalation", "compliance"): [
                "Escalating to Level {level} review per SAR requirements. Customer exhibits structured deposit pattern: "
                "{count} deposits totaling ${amount} over {days} days, each below reporting threshold. "
                "Enhanced due diligence required before account activity can resume.",
            ],
            ("exception", "trading"): [
                "Exception granted for position limit override. Client {client} approved for increase from "
                "${current}M to ${new}M in {sector} sector. Risk Committee approval obtained. "
                "Additional margin of ${margin}M posted as collateral. Valid for {days} days.",
            ],
        }

        with self.driver.session(database=self.database) as session:
            for i in range(NUM_DECISIONS):
                decision_id = str(uuid.uuid4())
                self.decision_ids.append(decision_id)

                # Select decision type with weighted probability
                decision_type = random.choices(
                    [d[0] for d in DECISION_TYPES],
                    weights=[d[1] for d in DECISION_TYPES],
                )[0]
                category = random.choice(DECISION_CATEGORIES)

                # Generate reasoning
                key = (decision_type, category)
                if key in reasoning_templates:
                    template = random.choice(reasoning_templates[key])
                    reasoning = template.format(
                        score=random.randint(580, 820),
                        income=random.randint(40000, 250000),
                        dti=random.randint(15, 55),
                        months=random.randint(6, 120),
                        limit=random.randint(5000, 100000),
                        marks=random.randint(1, 5),
                        reason=random.choice(
                            ["velocity check", "new device", "unusual amount", "new payee"]
                        ),
                        method=random.choice(["phone callback", "SMS OTP", "knowledge-based auth"]),
                        amount=random.randint(500, 50000),
                        code=random.randint(1000, 9999),
                        count=random.randint(3, 15),
                        minutes=random.randint(2, 30),
                        location1=fake.city(),
                        location2=fake.city(),
                        level=random.randint(1, 3),
                        days=random.randint(10, 90),
                        client=fake.company(),
                        current=random.randint(5, 50),
                        new=random.randint(50, 200),
                        sector=random.choice(["Technology", "Healthcare", "Energy", "Finance"]),
                        margin=random.randint(1, 20),
                    )
                else:
                    reasoning = f"Decision made for {category} case. Type: {decision_type}. Standard review completed with confidence score of {random.uniform(0.7, 0.99):.2f}."

                # Risk factors
                risk_factors = random.sample(
                    [
                        "high_amount",
                        "new_account",
                        "unusual_pattern",
                        "velocity_trigger",
                        "geographic_anomaly",
                        "new_device",
                        "after_hours",
                        "round_amount",
                        "multiple_beneficiaries",
                        "high_risk_country",
                    ],
                    k=random.randint(0, 4),
                )

                # Status based on decision type
                status_map = {
                    "approval": "approved",
                    "rejection": "rejected",
                    "escalation": "escalated",
                    "exception": "approved",
                    "override": "approved",
                    "review": "completed",
                }

                # Link to entities
                about_person = random.choice(self.person_ids) if random.random() > 0.3 else None
                about_account = random.choice(self.account_ids) if random.random() > 0.4 else None
                made_by = random.choice(self.employee_ids)
                applied_policy = random.choice(self.policy_ids) if random.random() > 0.5 else None

                session.run(
                    """
                    CREATE (d:Decision {
                        id: $id,
                        decision_type: $decision_type,
                        category: $category,
                        status: $status,
                        decision_timestamp: datetime($timestamp),
                        reasoning: $reasoning,
                        reasoning_summary: $summary,
                        confidence_score: $confidence,
                        risk_factors: $risk_factors,
                        source_system: $source,
                        created_at: datetime()
                    })
                    WITH d
                    OPTIONAL MATCH (p:Person {id: $about_person})
                    OPTIONAL MATCH (a:Account {id: $about_account})
                    OPTIONAL MATCH (e:Employee {id: $made_by})
                    OPTIONAL MATCH (pol:Policy {id: $applied_policy})
                    FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END | CREATE (d)-[:ABOUT]->(p))
                    FOREACH (_ IN CASE WHEN a IS NOT NULL THEN [1] ELSE [] END | CREATE (d)-[:ABOUT]->(a))
                    FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END | CREATE (d)-[:MADE_BY]->(e))
                    FOREACH (_ IN CASE WHEN pol IS NOT NULL THEN [1] ELSE [] END | CREATE (d)-[:APPLIED_POLICY]->(pol))
                    """,
                    {
                        "id": decision_id,
                        "decision_type": decision_type,
                        "category": category,
                        "status": status_map.get(decision_type, "completed"),
                        "timestamp": fake.date_time_between(
                            start_date="-2y", end_date="now"
                        ).isoformat(),
                        "reasoning": reasoning,
                        "summary": reasoning[:100] + "..." if len(reasoning) > 100 else reasoning,
                        "confidence": round(random.uniform(0.65, 0.99), 3),
                        "risk_factors": risk_factors,
                        "source": random.choice(["CRM", "Trading", "Compliance", "Risk"]),
                        "about_person": about_person,
                        "about_account": about_account,
                        "made_by": made_by,
                        "applied_policy": applied_policy,
                    },
                )
        print(f"Created {NUM_DECISIONS} decisions.")

    def generate_alerts(self):
        """Generate alert nodes linked to transactions and accounts."""
        print(f"Generating {NUM_ALERTS} alerts...")
        alert_types = [
            "fraud_detection",
            "velocity_check",
            "geographic_anomaly",
            "amount_threshold",
            "pattern_match",
            "sanctions_hit",
            "device_change",
            "behavioral_anomaly",
        ]
        severities = ["low", "medium", "high", "critical"]
        statuses = ["open", "investigating", "resolved", "false_positive", "escalated"]

        with self.driver.session(database=self.database) as session:
            for i in range(NUM_ALERTS):
                alert_id = str(uuid.uuid4())
                self.alert_ids.append(alert_id)

                alert_type = random.choice(alert_types)
                severity = random.choices(severities, weights=[0.3, 0.4, 0.2, 0.1])[0]
                status = random.choices(statuses, weights=[0.15, 0.2, 0.4, 0.15, 0.1])[0]

                # Link to transaction or account
                triggered_by_txn = (
                    random.choice(self.transaction_ids) if random.random() > 0.3 else None
                )
                triggered_by_account = (
                    random.choice(self.account_ids) if random.random() > 0.5 else None
                )
                assigned_to = random.choice(self.employee_ids) if random.random() > 0.4 else None
                resolved_by_decision = (
                    random.choice(self.decision_ids)
                    if status in ["resolved", "false_positive"] and random.random() > 0.3
                    else None
                )

                session.run(
                    """
                    CREATE (a:Alert {
                        id: $id,
                        alert_number: $alert_number,
                        alert_type: $alert_type,
                        severity: $severity,
                        status: $status,
                        description: $description,
                        triggered_at: datetime($triggered_at),
                        resolved_at: $resolved_at,
                        risk_score: $risk_score,
                        source_system: $source_system,
                        created_at: datetime()
                    })
                    WITH a
                    OPTIONAL MATCH (t:Transaction {id: $triggered_by_txn})
                    OPTIONAL MATCH (acc:Account {id: $triggered_by_account})
                    OPTIONAL MATCH (e:Employee {id: $assigned_to})
                    OPTIONAL MATCH (d:Decision {id: $resolved_by_decision})
                    FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END | CREATE (a)-[:TRIGGERED_BY]->(t))
                    FOREACH (_ IN CASE WHEN acc IS NOT NULL THEN [1] ELSE [] END | CREATE (a)-[:REGARDING]->(acc))
                    FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END | CREATE (a)-[:ASSIGNED_TO]->(e))
                    FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END | CREATE (a)-[:RESOLVED_BY]->(d))
                    """,
                    {
                        "id": alert_id,
                        "alert_number": f"ALT{str(i + 1).zfill(6)}",
                        "alert_type": alert_type,
                        "severity": severity,
                        "status": status,
                        "description": f"{alert_type.replace('_', ' ').title()} alert: {fake.sentence(nb_words=8)}",
                        "triggered_at": fake.date_time_between(
                            start_date="-1y", end_date="now"
                        ).isoformat(),
                        "resolved_at": fake.date_time_between(
                            start_date="-6m", end_date="now"
                        ).isoformat()
                        if status in ["resolved", "false_positive"]
                        else None,
                        "risk_score": round(random.uniform(0.3, 1.0), 3),
                        "source_system": random.choice(
                            ["Fraud Detection", "AML", "Compliance", "Risk Engine"]
                        ),
                        "triggered_by_txn": triggered_by_txn,
                        "triggered_by_account": triggered_by_account,
                        "assigned_to": assigned_to,
                        "resolved_by_decision": resolved_by_decision,
                    },
                )
        print(f"Created {NUM_ALERTS} alerts.")

    def generate_support_tickets(self):
        """Generate support ticket nodes linked to persons and accounts."""
        print(f"Generating {NUM_SUPPORT_TICKETS} support tickets...")
        ticket_types = [
            "account_inquiry",
            "transaction_dispute",
            "fraud_report",
            "password_reset",
            "statement_request",
            "limit_increase",
            "complaint",
            "general_inquiry",
        ]
        priorities = ["low", "medium", "high", "urgent"]
        statuses = ["open", "in_progress", "pending_customer", "resolved", "closed"]
        channels = ["phone", "email", "chat", "branch", "mobile_app"]

        with self.driver.session(database=self.database) as session:
            for i in range(NUM_SUPPORT_TICKETS):
                ticket_id = str(uuid.uuid4())
                self.support_ticket_ids.append(ticket_id)

                ticket_type = random.choice(ticket_types)
                priority = random.choices(priorities, weights=[0.3, 0.4, 0.2, 0.1])[0]
                status = random.choices(statuses, weights=[0.1, 0.15, 0.1, 0.35, 0.3])[0]

                # Link to person and optionally account
                submitted_by = random.choice(self.person_ids)
                regarding_account = (
                    random.choice(self.account_ids) if random.random() > 0.4 else None
                )
                assigned_to = random.choice(self.employee_ids) if random.random() > 0.3 else None
                related_decision = (
                    random.choice(self.decision_ids)
                    if status in ["resolved", "closed"] and random.random() > 0.5
                    else None
                )
                related_transaction = (
                    random.choice(self.transaction_ids)
                    if ticket_type == "transaction_dispute" and random.random() > 0.3
                    else None
                )

                session.run(
                    """
                    CREATE (s:SupportTicket {
                        id: $id,
                        ticket_number: $ticket_number,
                        ticket_type: $ticket_type,
                        priority: $priority,
                        status: $status,
                        subject: $subject,
                        description: $description,
                        channel: $channel,
                        submitted_at: datetime($submitted_at),
                        resolved_at: $resolved_at,
                        satisfaction_score: $satisfaction_score,
                        source_system: 'Support',
                        created_at: datetime()
                    })
                    WITH s
                    MATCH (p:Person {id: $submitted_by})
                    CREATE (s)-[:SUBMITTED_BY]->(p)
                    WITH s
                    OPTIONAL MATCH (acc:Account {id: $regarding_account})
                    OPTIONAL MATCH (e:Employee {id: $assigned_to})
                    OPTIONAL MATCH (d:Decision {id: $related_decision})
                    OPTIONAL MATCH (t:Transaction {id: $related_transaction})
                    FOREACH (_ IN CASE WHEN acc IS NOT NULL THEN [1] ELSE [] END | CREATE (s)-[:REGARDING]->(acc))
                    FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END | CREATE (s)-[:ASSIGNED_TO]->(e))
                    FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END | CREATE (s)-[:RESOLVED_BY]->(d))
                    FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END | CREATE (s)-[:RELATED_TO]->(t))
                    """,
                    {
                        "id": ticket_id,
                        "ticket_number": f"TKT{str(i + 1).zfill(6)}",
                        "ticket_type": ticket_type,
                        "priority": priority,
                        "status": status,
                        "subject": f"{ticket_type.replace('_', ' ').title()}: {fake.sentence(nb_words=5)}",
                        "description": fake.paragraph(nb_sentences=3),
                        "channel": random.choice(channels),
                        "submitted_at": fake.date_time_between(
                            start_date="-1y", end_date="now"
                        ).isoformat(),
                        "resolved_at": fake.date_time_between(
                            start_date="-6m", end_date="now"
                        ).isoformat()
                        if status in ["resolved", "closed"]
                        else None,
                        "satisfaction_score": random.randint(1, 5)
                        if status in ["resolved", "closed"]
                        else None,
                        "submitted_by": submitted_by,
                        "regarding_account": regarding_account,
                        "assigned_to": assigned_to,
                        "related_decision": related_decision,
                        "related_transaction": related_transaction,
                    },
                )
        print(f"Created {NUM_SUPPORT_TICKETS} support tickets.")

    def create_causal_chains(self):
        """Create causal relationships between decisions."""
        print("Creating causal chains between decisions...")
        with self.driver.session(database=self.database) as session:
            # Create some CAUSED relationships
            for _ in range(int(NUM_DECISIONS * 0.15)):
                d1 = random.choice(self.decision_ids)
                d2 = random.choice(self.decision_ids)
                if d1 != d2:
                    session.run(
                        """
                        MATCH (d1:Decision {id: $d1})
                        MATCH (d2:Decision {id: $d2})
                        WHERE d1.decision_timestamp < d2.decision_timestamp
                        MERGE (d1)-[:CAUSED {
                            confidence: $confidence,
                            causation_type: $type
                        }]->(d2)
                        """,
                        {
                            "d1": d1,
                            "d2": d2,
                            "confidence": round(random.uniform(0.6, 1.0), 2),
                            "type": random.choice(["direct", "contributing", "enabling"]),
                        },
                    )

            # Create some INFLUENCED relationships
            for _ in range(int(NUM_DECISIONS * 0.2)):
                d1 = random.choice(self.decision_ids)
                d2 = random.choice(self.decision_ids)
                if d1 != d2:
                    session.run(
                        """
                        MATCH (d1:Decision {id: $d1})
                        MATCH (d2:Decision {id: $d2})
                        WHERE d1.decision_timestamp < d2.decision_timestamp
                        MERGE (d1)-[:INFLUENCED {
                            weight: $weight,
                            influence_type: $type
                        }]->(d2)
                        """,
                        {
                            "d1": d1,
                            "d2": d2,
                            "weight": round(random.uniform(0.3, 1.0), 2),
                            "type": random.choice(["precedent", "policy", "context"]),
                        },
                    )

            # Create some PRECEDENT_FOR relationships
            for _ in range(int(NUM_DECISIONS * 0.1)):
                d1 = random.choice(self.decision_ids)
                d2 = random.choice(self.decision_ids)
                if d1 != d2:
                    session.run(
                        """
                        MATCH (d1:Decision {id: $d1})
                        MATCH (d2:Decision {id: $d2})
                        WHERE d1.decision_timestamp < d2.decision_timestamp
                        MERGE (d1)-[:PRECEDENT_FOR {
                            similarity_score: $similarity,
                            outcome_relevance: $relevance
                        }]->(d2)
                        """,
                        {
                            "d1": d1,
                            "d2": d2,
                            "similarity": round(random.uniform(0.5, 1.0), 2),
                            "relevance": round(random.uniform(0.4, 1.0), 2),
                        },
                    )
        print("Created causal chains.")

    def generate_all(self):
        """Generate all sample data."""
        print("\n" + "=" * 50)
        print("CONTEXT GRAPH SAMPLE DATA GENERATOR")
        print("=" * 50 + "\n")

        self.clear_database()
        self.create_constraints_and_indexes()
        self.generate_organizations()
        self.generate_employees()
        self.generate_policies()
        self.generate_persons()
        self.generate_accounts()
        self.generate_transactions()
        self.generate_decisions()
        self.generate_alerts()
        self.generate_support_tickets()
        self.create_causal_chains()
        self.connect_organizations()

        print("\n" + "=" * 50)
        print("DATA GENERATION COMPLETE!")
        print("=" * 50)
        print(f"\nGenerated:")
        print(
            f"  - {NUM_ORGANIZATIONS} organizations (connected to persons, accounts, transactions)"
        )
        print(f"  - {NUM_EMPLOYEES} employees")
        print(f"  - {NUM_POLICIES} policies")
        print(f"  - {NUM_PERSONS} persons")
        print(f"  - {NUM_ACCOUNTS} accounts")
        print(f"  - {NUM_TRANSACTIONS} transactions")
        print(f"  - {NUM_DECISIONS} decisions (with causal chains)")
        print(f"  - {NUM_ALERTS} alerts")
        print(f"  - {NUM_SUPPORT_TICKETS} support tickets")
        print("\nYou can now run GDS algorithms to generate FastRP embeddings.")


if __name__ == "__main__":
    generator = DataGenerator()
    try:
        generator.generate_all()
    finally:
        generator.close()
