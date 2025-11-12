import type { BehaviorHandler } from "./BehaviorHandler";
import { ShortBlurbWithList } from "./ShortBlurbWithList";
import { LongFormAnswer } from "./LongFormAnswer";
import { ShortAnswer } from "./ShortAnswer";
import { DetailedExplanation } from "./DetailedExplanation";
import { SinglePageSummary } from "./SinglePageSummary";
import { Comparison } from "./Comparison";
import { CollectionOverview } from "./CollectionOverview";
import { RecentItems } from "./RecentItems";
import { NavigationHelp } from "./NavigationHelp";

/**
 * Behavior Registry
 *
 * Central registry of all available response behaviors.
 * Behaviors are instantiated once and reused across requests.
 */
class BehaviorRegistry {
  private behaviors: Map<string, BehaviorHandler>;
  private defaultBehavior: BehaviorHandler;

  constructor() {
    this.behaviors = new Map();

    // Register all behaviors
    this.registerBehavior(new ShortBlurbWithList());
    this.registerBehavior(new LongFormAnswer());
    this.registerBehavior(new ShortAnswer());
    this.registerBehavior(new DetailedExplanation());
    this.registerBehavior(new SinglePageSummary());
    this.registerBehavior(new Comparison());
    this.registerBehavior(new CollectionOverview());
    this.registerBehavior(new RecentItems());
    this.registerBehavior(new NavigationHelp());

    // Default behavior is LongFormAnswer
    this.defaultBehavior = new LongFormAnswer();
  }

  /**
   * Register a behavior handler in the registry
   */
  private registerBehavior(handler: BehaviorHandler): void {
    this.behaviors.set(handler.name, handler);

    // Also register with underscores replaced by hyphens for flexibility
    const alternativeName = handler.name.replace(/_/g, "-");
    if (alternativeName !== handler.name) {
      this.behaviors.set(alternativeName, handler);
    }
  }

  /**
   * Get a behavior by name, or return default if not found
   */
  getBehavior(name: string | undefined): BehaviorHandler {
    if (!name) {
      return this.defaultBehavior;
    }

    // Normalize name (lowercase, handle both _ and -)
    const normalizedName = name.toLowerCase().trim();
    const behavior = this.behaviors.get(normalizedName);

    if (behavior) {
      return behavior;
    }

    // Try alternative formats
    const withUnderscores = normalizedName.replace(/-/g, "_");
    const withHyphens = normalizedName.replace(/_/g, "-");

    return (
      this.behaviors.get(withUnderscores) ||
      this.behaviors.get(withHyphens) ||
      this.defaultBehavior
    );
  }

  /**
   * Get all registered behavior names
   */
  getAllBehaviorNames(): string[] {
    const uniqueNames = new Set<string>();
    for (const [name, handler] of this.behaviors.entries()) {
      uniqueNames.add(handler.name); // Use canonical name from handler
    }
    return Array.from(uniqueNames).sort();
  }

  /**
   * Check if a behavior exists
   */
  hasBehavior(name: string): boolean {
    const normalizedName = name.toLowerCase().trim();
    return (
      this.behaviors.has(normalizedName) ||
      this.behaviors.has(normalizedName.replace(/-/g, "_")) ||
      this.behaviors.has(normalizedName.replace(/_/g, "-"))
    );
  }
}

// Singleton instance
const registry = new BehaviorRegistry();

/**
 * Get a behavior handler by name.
 * Returns default behavior if name is invalid or not found.
 *
 * @param name - Behavior name (e.g., "short_blurb_with_list" or "short-blurb-with-list")
 * @returns BehaviorHandler instance
 */
export function getBehavior(name: string | undefined): BehaviorHandler {
  return registry.getBehavior(name);
}

/**
 * Get all available behavior names
 */
export function getAllBehaviors(): string[] {
  return registry.getAllBehaviorNames();
}

/**
 * Check if a behavior exists
 */
export function behaviorExists(name: string): boolean {
  return registry.hasBehavior(name);
}

// Export types
export type { BehaviorHandler, BehaviorContext, BehaviorResponse, CustomIntent } from "./BehaviorHandler";
