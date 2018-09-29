import unittest
import os

from app import app, db

TEST_DB = 'test.db'


class BasicTestCase(unittest.TestCase):
    def test_index(self):
        """initial test. ensure flask was set up correctly"""
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_database(self):
        """initial test. ensure that the database exists"""
        tester = os.path.exists("users.db")
        self.assertTrue(tester)


class AppTestCase(unittest.TestCase):
    def setUp(self):
        """Set up a blank temp database before each test"""
        basedir = os.path.abspath(os.path.dirname(__file__))
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
            os.path.join(basedir, TEST_DB)
        self.app = app.test_client()
        db.create_all()

    def tearDown(self):
        """Destroy blank temp database after each test"""
        db.drop_all()

    def login(self, username, password):
        """Login helper function"""
        return self.app.post(
            '/login', data=dict(username=username, password=password), follow_redirects=True
        )

    def logout(self):
        """Logout helper function"""
        return self.app.get('/logout', follow_redirects=True)

    # assert functions

    def test_empty_db(self):
        """Ensure database is blank"""
        rv = self.app.get('/')
        self.assertIn(b'No entries yet. Add some!', rv.data)


if __name__ == '__main__':
    unittest.main()
