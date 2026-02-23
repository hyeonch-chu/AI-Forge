// Jest configuration for the AI-Forge Next.js UI.
// Uses next/jest to auto-configure transforms, module mappers, and CSS handling
// so that Next.js-specific features work in the test environment.

const nextJest = require('next/jest');

// Provide the path to the Next.js app so next/jest can load next.config.js and .env files
const createJestConfig = nextJest({ dir: './' });

/** @type {import('jest').Config} */
const customConfig = {
  // Use jsdom to simulate a browser environment for React component tests
  testEnvironment: 'jest-environment-jsdom',

  // Run @testing-library/jest-dom matchers setup after the framework loads
  setupFilesAfterFramework: ['<rootDir>/jest.setup.ts'],

  // Resolve @/* imports to src/*
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },

  // Only discover tests under src/__tests__
  testMatch: ['<rootDir>/src/__tests__/**/*.{test,spec}.{ts,tsx}'],
};

// createJestConfig merges Next.js defaults (babel-jest transform, CSS mocks, etc.)
// with our custom configuration
module.exports = createJestConfig(customConfig);
